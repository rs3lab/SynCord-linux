/*
 * Queued spinlock
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * (C) Copyright 2013-2015 Hewlett-Packard Development Company, L.P.
 * (C) Copyright 2013-2014,2018 Red Hat, Inc.
 * (C) Copyright 2015 Intel Corp.
 * (C) Copyright 2015 Hewlett-Packard Enterprise Development LP
 *
 * Authors: Waiman Long <longman@redhat.com>
 *          Peter Zijlstra <peterz@infradead.org>
 */

#ifndef _GEN_PV_LOCK_SLOWPATH

#include <linux/smp.h>
#include <linux/bug.h>
#include <linux/cpumask.h>
#include <linux/percpu.h>
#include <linux/hardirq.h>
#include <linux/mutex.h>
#include <linux/prefetch.h>
#include <asm/byteorder.h>
#include <asm/qspinlock.h>
#include <linux/random.h>

/*
 * Include queued spinlock statistics code
 */
#include "qspinlock_stat.h"

/*
 * The basic principle of a queue-based spinlock can best be understood
 * by studying a classic queue-based spinlock implementation called the
 * MCS lock. The paper below provides a good description for this kind
 * of lock.
 *
 * http://www.cise.ufl.edu/tr/DOC/REP-1992-71.pdf
 *
 * This queued spinlock implementation is based on the MCS lock, however to make
 * it fit the 4 bytes we assume spinlock_t to be, and preserve its existing
 * API, we must modify it somehow.
 *
 * In particular; where the traditional MCS lock consists of a tail pointer
 * (8 bytes) and needs the next pointer (another 8 bytes) of its own node to
 * unlock the next pending (next->locked), we compress both these: {tail,
 * next->locked} into a single u32 value.
 *
 * Since a spinlock disables recursion of its own context and there is a limit
 * to the contexts that can nest; namely: task, softirq, hardirq, nmi. As there
 * are at most 4 nesting levels, it can be encoded by a 2-bit number. Now
 * we can encode the tail by combining the 2-bit nesting level with the cpu
 * number. With one byte for the lock value and 3 bytes for the tail, only a
 * 32-bit word is now needed. Even though we only need 1 bit for the lock,
 * we extend it to a full byte to achieve better performance for architectures
 * that support atomic byte write.
 *
 * We also change the first spinner to spin on the lock bit instead of its
 * node; whereby avoiding the need to carry a node from lock to unlock, and
 * preserving existing lock API. This also makes the unlock code simpler and
 * faster.
 *
 * N.B. The current implementation only supports architectures that allow
 *      atomic operations on smaller 8-bit and 16-bit data types.
 *
 */

#include "mcs_spinlock.h"

#ifdef CONFIG_PARAVIRT_SPINLOCKS
#define MAX_NODES   8
#else
#define MAX_NODES   4
#endif

/*
 * The pending bit spinning loop count.
 * This heuristic is used to limit the number of lockword accesses
 * made by atomic_cond_read_relaxed when waiting for the lock to
 * transition out of the "== _Q_PENDING_VAL" state. We don't spin
 * indefinitely because there's no guarantee that we'll make forward
 * progress.
 */
#ifndef _Q_PENDING_LOOPS
#define _Q_PENDING_LOOPS	1
#endif

/*
 * Per-CPU queue node structures; we can never have more than 4 nested
 * contexts: task, softirq, hardirq, nmi.
 *
 * Exactly fits two 64-byte cachelines on a 64-bit architecture.
 *
 * PV adds more storage for PV state, and thus needs three cachelines.
 */
static DEFINE_PER_CPU_ALIGNED(struct mcs_spinlock, mcs_nodes[MAX_NODES]);

/* Per-CPU pseudo-random number seed */
static DEFINE_PER_CPU(u32, seed);

/*
 * Controls the probability for intra-socket lock hand-off. It can be
 * tuned and depend, e.g., on the number of CPUs per socket. For now,
 * choose a value that provides reasonable long-term fairness without
 * sacrificing performance compared to a version that does not have any
 * fairness guarantees.
 */
#define INTRA_SOCKET_HANDOFF_PROB_ARG   0x10000

/*
 * We must be able to distinguish between no-tail and the tail at 0:0,
 * therefore increment the cpu number by one.
 */

static inline __pure u32 encode_tail(int cpu, int idx)
{
	u32 tail;

	tail  = (cpu + 1) << _Q_TAIL_CPU_OFFSET;
	tail |= idx << _Q_TAIL_IDX_OFFSET; /* assume < 4 */

	return tail;
}

static inline __pure struct mcs_spinlock *decode_tail(u32 tail)
{
	int cpu = (tail >> _Q_TAIL_CPU_OFFSET) - 1;
	int idx = (tail &  _Q_TAIL_IDX_MASK) >> _Q_TAIL_IDX_OFFSET;

	return per_cpu_ptr(&mcs_nodes[idx], cpu);
}

#define _Q_LOCKED_PENDING_MASK (_Q_LOCKED_MASK | _Q_PENDING_MASK)

#if _Q_PENDING_BITS == 8
/**
 * clear_pending - clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,* -> *,0,*
 */
static __always_inline void clear_pending(struct qspinlock *lock)
{
	WRITE_ONCE(lock->pending, 0);
}

/**
 * clear_pending_set_locked - take ownership and clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,0 -> *,0,1
 *
 * Lock stealing is not allowed if this function is used.
 */
static __always_inline void clear_pending_set_locked(struct qspinlock *lock)
{
	WRITE_ONCE(lock->locked_pending, _Q_LOCKED_VAL);
}

/*
 * xchg_tail - Put in the new queue tail code word & retrieve previous one
 * @lock : Pointer to queued spinlock structure
 * @tail : The new queue tail code word
 * Return: The previous queue tail code word
 *
 * xchg(lock, tail), which heads an address dependency
 *
 * p,*,* -> n,*,* ; prev = xchg(lock, node)
 */
static __always_inline u32 xchg_tail(struct qspinlock *lock, u32 tail)
{
	/*
	 * We can use relaxed semantics since the caller ensures that the
	 * MCS node is properly initialized before updating the tail.
	 */
	return (u32)xchg_relaxed(&lock->tail,
				 tail >> _Q_TAIL_OFFSET) << _Q_TAIL_OFFSET;
}

#else /* _Q_PENDING_BITS == 8 */

/**
 * clear_pending - clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,* -> *,0,*
 */
static __always_inline void clear_pending(struct qspinlock *lock)
{
	atomic_andnot(_Q_PENDING_VAL, &lock->val);
}

/**
 * clear_pending_set_locked - take ownership and clear the pending bit.
 * @lock: Pointer to queued spinlock structure
 *
 * *,1,0 -> *,0,1
 */
static __always_inline void clear_pending_set_locked(struct qspinlock *lock)
{
	atomic_add(-_Q_PENDING_VAL + _Q_LOCKED_VAL, &lock->val);
}

/**
 * xchg_tail - Put in the new queue tail code word & retrieve previous one
 * @lock : Pointer to queued spinlock structure
 * @tail : The new queue tail code word
 * Return: The previous queue tail code word
 *
 * xchg(lock, tail)
 *
 * p,*,* -> n,*,* ; prev = xchg(lock, node)
 */
static __always_inline u32 xchg_tail(struct qspinlock *lock, u32 tail)
{
	u32 old, new, val = atomic_read(&lock->val);

	for (;;) {
		new = (val & _Q_LOCKED_PENDING_MASK) | tail;
		/*
		 * We can use relaxed semantics since the caller ensures that
		 * the MCS node is properly initialized before updating the
		 * tail.
		 */
		old = atomic_cmpxchg_relaxed(&lock->val, val, new);
		if (old == val)
			break;

		val = old;
	}
	return old;
}
#endif /* _Q_PENDING_BITS == 8 */

/**
 * set_locked - Set the lock bit and own the lock
 * @lock: Pointer to queued spinlock structure
 *
 * *,*,0 -> *,0,1
 */
static __always_inline void set_locked(struct qspinlock *lock)
{
	WRITE_ONCE(lock->locked, _Q_LOCKED_VAL);
}


/*
 * Generate the native code for queued_spin_unlock_slowpath(); provide NOPs for
 * all the PV callbacks.
 */

static __always_inline void __pv_init_node(struct mcs_spinlock *node) { }
static __always_inline void __pv_wait_node(struct mcs_spinlock *node,
					   struct mcs_spinlock *prev) { }
static __always_inline void __pv_kick_node(struct qspinlock *lock,
					   struct mcs_spinlock *node) { }
static __always_inline u32  __pv_wait_head_or_lock(struct qspinlock *lock,
						   struct mcs_spinlock *node)
						   { return 0; }

#define pv_enabled()		false

#define pv_init_node		__pv_init_node
#define pv_wait_node		__pv_wait_node
#define pv_kick_node		__pv_kick_node
#define pv_wait_head_or_lock	__pv_wait_head_or_lock

#ifdef CONFIG_PARAVIRT_SPINLOCKS
#define queued_spin_lock_slowpath	native_queued_spin_lock_slowpath
#endif

#define MCS_NODE(ptr)	((struct mcs_spinlock *)(ptr))

static inline __pure int decode_socket(u32 socket_and_count)
{
	int socket = (socket_and_count >> _Q_SOCKET_OFFSET) - 1;

	return socket;
}

static inline __pure int decode_count(u32 socket_and_count)
{
	int count = socket_and_count & _Q_IDX_MASK;

	return count;
}

static inline void set_socket(struct mcs_spinlock *node, int socket)
{
	u32 val;

	val  = (socket + 1) << _Q_SOCKET_OFFSET;
	val |= decode_count(node->socket_and_count);

	node->socket_and_count = val;
}

static int inline cmp_func(struct mcs_spinlock * me, struct mcs_spinlock * succ)
{
	int my_socket;
	/* Get socket, which would not be set if we entered an empty queue. */
	my_socket = decode_socket(me->socket_and_count);
	if (my_socket == -1)
		my_socket = numa_cpu_node(smp_processor_id());

	return (decode_socket(succ->socket_and_count) == my_socket);
}

static struct mcs_spinlock *find_successor(struct mcs_spinlock *me)
{
	struct mcs_spinlock *head_other, *tail_other, *cur;

	struct mcs_spinlock *next = me->next;
	/* @next should be set, else we would not be calling this function. */
	WARN_ON_ONCE(next == NULL);


	/*
	 * Fast path - check whether the immediate successor runs on
	 * the same socket.
	 */
	if (cmp_func(me, next))
		return next;

	head_other = next;
	tail_other = next;

	/*
	 * Traverse the main waiting queue starting from the successor of my
	 * successor, and look for a thread running on the same socket.
	 */
	cur = READ_ONCE(next->next);
	while (cur) {
		if (cmp_func(me, cur)) {
			/*
			 * Found a thread on the same socket. Move threads
			 * between me and that node into the secondary queue.
			 */
			if (me->locked > 1)
				MCS_NODE(me->locked)->tail->next = head_other;
			else
				me->locked = (uintptr_t)head_other;
			tail_other->next = NULL;
			MCS_NODE(me->locked)->tail = tail_other;
			return cur;
		}
		tail_other = cur;
		cur = READ_ONCE(cur->next);
	}
	return NULL;
}

/*
 * xorshift function for generating pseudo-random numbers:
 * https://en.wikipedia.org/wiki/Xorshift
 */
static inline u32 xor_random(void)
{
	u32 v;

	v = this_cpu_read(seed);
	if (v == 0)
		get_random_bytes(&v, sizeof(u32));

	v ^= v << 6;
	v ^= v >> 21;
	v ^= v << 7;
	this_cpu_write(seed, v);

	return v;
}

/*
 * Return false with probability 1 / @range.
 * @range must be a power of 2.
 */
static bool probably(unsigned int range)
{
	return xor_random() & (range - 1);
}

#endif /* _GEN_PV_LOCK_SLOWPATH */

/**
 * queued_spin_lock_slowpath - acquire the queued spinlock
 * @lock: Pointer to queued spinlock structure
 * @val: Current value of the queued spinlock 32-bit word
 *
 * (queue tail, pending bit, lock value)
 *
 *              fast     :    slow                                  :    unlock
 *                       :                                          :
 * uncontended  (0,0,0) -:--> (0,0,1) ------------------------------:--> (*,*,0)
 *                       :       | ^--------.------.             /  :
 *                       :       v           \      \            |  :
 * pending               :    (0,1,1) +--> (0,1,0)   \           |  :
 *                       :       | ^--'              |           |  :
 *                       :       v                   |           |  :
 * uncontended           :    (n,x,y) +--> (n,0,0) --'           |  :
 *   queue               :       | ^--'                          |  :
 *                       :       v                               |  :
 * contended             :    (*,x,y) +--> (*,0,0) ---> (*,0,1) -'  :
 *   queue               :         ^--'                             :
 */
void queued_spin_lock_slowpath(struct qspinlock *lock, u32 val)
{
	struct mcs_spinlock *prev, *next, *node, *succ;
	u32 old, tail, new;
	int idx, cpuid;

	BUILD_BUG_ON(CONFIG_NR_CPUS >= (1U << _Q_TAIL_CPU_BITS));

	/* if (pv_enabled()) */
	/*     goto pv_queue; */

	/* if (virt_spin_lock(lock)) */
	/*     return; */

	/*
	 * Wait for in-progress pending->locked hand-overs with a bounded
	 * number of spins so that we guarantee forward progress.
	 *
	 * 0,1,0 -> 0,0,1
	 */
	if (val == _Q_PENDING_VAL) {
		int cnt = _Q_PENDING_LOOPS;
		val = atomic_cond_read_relaxed(&lock->val,
				(VAL != _Q_PENDING_VAL) || !cnt--);
	}

	/*
	 * If we observe any contention; queue.
	 */
	if (val & ~_Q_LOCKED_MASK)
		goto queue;

	/*
	 * trylock || pending
	 *
	 * 0,0,0 -> 0,0,1 ; trylock
	 * 0,0,1 -> 0,1,1 ; pending
	 */
	val = atomic_fetch_or_acquire(_Q_PENDING_VAL, &lock->val);
	if (!(val & ~_Q_LOCKED_MASK)) {
		/*
		 * We're pending, wait for the owner to go away.
		 *
		 * *,1,1 -> *,1,0
		 *
		 * this wait loop must be a load-acquire such that we match the
		 * store-release that clears the locked bit and create lock
		 * sequentiality; this is because not all
		 * clear_pending_set_locked() implementations imply full
		 * barriers.
		 */
		if (val & _Q_LOCKED_MASK) {
			atomic_cond_read_acquire(&lock->val,
					!(VAL & _Q_LOCKED_MASK));
		}

		/*
		 * take ownership and clear the pending bit.
		 *
		 * *,1,0 -> *,0,1
		 */
		clear_pending_set_locked(lock);
		return;
	}

	/*
	 * If pending was clear but there are waiters in the queue, then
	 * we need to undo our setting of pending before we queue ourselves.
	 */
	if (!(val & _Q_PENDING_MASK))
		clear_pending(lock);

	/*
	 * End of pending bit optimistic spinning and beginning of MCS
	 * queuing.
	 */
queue:
pv_queue:
	node = this_cpu_ptr(&mcs_nodes[0]);
#ifdef CONFIG_DEBUG_SPINLOCK
	BUG_ON(decode_count(node->socket_and_count) >= 3);
#endif
	idx = decode_count(node->socket_and_count++);
	cpuid = smp_processor_id();
	tail = encode_tail(cpuid, idx);

	node += idx;


	/*
	 * Ensure that we increment the head node->count before initialising
	 * the actual node. If the compiler is kind enough to reorder these
	 * stores, then an IRQ could overwrite our assignments.
	 */
	barrier();

	node->locked = 0;
	node->next = NULL;
	set_socket(node, -1);
	node->encoded_tail = tail;
	pv_init_node(node);

	/*
	 * We touched a (possibly) cold cacheline in the per-cpu queue node;
	 * attempt the trylock once more in the hope someone let go while we
	 * weren't watching.
	 */
	if (queued_spin_trylock(lock))
		goto release;

	/*
	 * Ensure that the initialisation of @node is complete before we
	 * publish the updated tail via xchg_tail() and potentially link
	 * @node into the waitqueue via WRITE_ONCE(prev->next, node) below.
	 */
	smp_wmb();

	/*
	 * Publish the updated tail.
	 * We have already touched the queueing cacheline; don't bother with
	 * pending stuff.
	 *
	 * p,*,* -> n,*,*
	 */
	old = xchg_tail(lock, tail);
	next = NULL;

	/*
	 * if there was a previous node; link it and wait until reaching the
	 * head of the waitqueue.
	 */
	if (old & _Q_TAIL_MASK) {
		prev = decode_tail(old);

		/*
		 * An explicit barrier after the store to @socket
		 * is not required as making the socket value visible is
		 * required only for performance, not correctness, and
		 * we rather avoid the cost of the barrier.
		 */
		set_socket(node, numa_cpu_node(cpuid));

		/* Link @node into the waitqueue. */
		WRITE_ONCE(prev->next, node);

		pv_wait_node(node, prev);
		arch_mcs_spin_lock_contended(&node->locked);

		/*
		 * While waiting for the MCS lock, the next pointer may have
		 * been set by another lock waiter. We optimistically load
		 * the next pointer & prefetch the cacheline for writing
		 * to reduce latency in the upcoming MCS unlock operation.
		 */
		next = READ_ONCE(node->next);
		if (next)
			prefetchw(next);
	} else {
		/* Must pass a non-zero value to successor when we unlock. */
		node->locked = 1;
	}

	/*
	 * we're at the head of the waitqueue, wait for the owner & pending to
	 * go away.
	 *
	 * *,x,y -> *,0,0
	 *
	 * this wait loop must use a load-acquire such that we match the
	 * store-release that clears the locked bit and create lock
	 * sequentiality; this is because the set_locked() function below
	 * does not imply a full barrier.
	 *
	 * The PV pv_wait_head_or_lock function, if active, will acquire
	 * the lock and return a non-zero value. So we have to skip the
	 * atomic_cond_read_acquire() call. As the next PV queue head hasn't
	 * been designated yet, there is no way for the locked value to become
	 * _Q_SLOW_VAL. So both the set_locked() and the
	 * atomic_cmpxchg_relaxed() calls will be safe.
	 *
	 * If PV isn't active, 0 will be returned instead.
	 *
	 */
	if ((val = pv_wait_head_or_lock(lock, node)))
		goto locked;

	val = atomic_cond_read_acquire(&lock->val, !(VAL & _Q_LOCKED_PENDING_MASK));

locked:
	/*
	 * claim the lock:
	 *
	 * n,0,0 -> 0,0,1 : lock, uncontended
	 * *,*,0 -> *,*,1 : lock, contended
	 *
	 * If the queue head is the only one in the queue (lock value == tail)
	 * and nobody is pending, clear the tail code and grab the lock.
	 * Otherwise, we only need to grab the lock.
	 */

	/*
	 * In the PV case we might already have _Q_LOCKED_VAL set.
	 *
	 * The atomic_cond_read_acquire() call above has provided the
	 * necessary acquire semantics required for locking.
	 */
	if ((val & _Q_TAIL_MASK) == tail) {
		/* Check whether the secondary queue is empty. */
		if (node->locked == 1) {
			if (atomic_try_cmpxchg_relaxed(&lock->val, &val,
						_Q_LOCKED_VAL))
				goto release; /* No contention */
		} else {
			/*
			 * Pass the lock to the first thread in the secondary
			 * queue, but first try to update the queue's tail to
			 * point to the last node in the secondary queue.
			 */
			succ = MCS_NODE(node->locked);
			new = succ->tail->encoded_tail + _Q_LOCKED_VAL;
			if (atomic_try_cmpxchg_relaxed(&lock->val, &val, new)) {
				arch_mcs_spin_unlock_contended(&succ->locked, 1);
				goto release;
			}
		}
	}
	/* Either somebody is queued behind us or _Q_PENDING_VAL is set */
	set_locked(lock);

	/*
	 * contended path; wait for next if not observed yet, release.
	 */
	if (!next)
		next = smp_cond_load_relaxed(&node->next, (VAL));

	/*
	 * Try to pass the lock to a thread running on the same socket.
	 * For long-term fairness, search for such a thread with high
	 * probability rather than always.
	 */
	succ = NULL;
	if (probably(INTRA_SOCKET_HANDOFF_PROB_ARG))
		succ = find_successor(node);

	if (succ) {
		arch_mcs_spin_unlock_contended(&succ->locked, node->locked);
	} else if (node->locked > 1) {
		/*
		 * If the secondary queue is not empty, pass the lock
		 * to the first node in that queue.
		 */
		succ = MCS_NODE(node->locked);
		succ->tail->next = next;
		arch_mcs_spin_unlock_contended(&succ->locked, 1);
	} else {
		/*
		 * Otherwise, pass the lock to the immediate successor
		 * in the main queue.
		 */
		arch_mcs_spin_unlock_contended(&next->locked, 1);
	}
	pv_kick_node(lock, next);

release:
	/*
	 * release the node
	 */
	__this_cpu_dec(mcs_nodes[0].socket_and_count);
}
EXPORT_SYMBOL(queued_spin_lock_slowpath);

/*
 * Generate the paravirt code for queued_spin_unlock_slowpath().
 */
#if !defined(_GEN_PV_LOCK_SLOWPATH) && defined(CONFIG_PARAVIRT_SPINLOCKS)
#define _GEN_PV_LOCK_SLOWPATH

#undef  pv_enabled
#define pv_enabled()	true

#undef pv_init_node
#undef pv_wait_node
#undef pv_kick_node
#undef pv_wait_head_or_lock

#undef  queued_spin_lock_slowpath
#define queued_spin_lock_slowpath	__pv_queued_spin_lock_slowpath

#include "qspinlock_paravirt.h"
#include "qspinlock.c"

#endif
