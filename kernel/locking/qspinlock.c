// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Queued spinlock
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
#include <linux/topology.h>
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
#define MAX_NODES	4

/*
 * On 64-bit architectures, the mcs_spinlock structure will be 16 bytes in
 * size and four of them will fit nicely in one 64-byte cacheline. For
 * pvqspinlock, however, we need more space for extra data. To accommodate
 * that, we insert two more long words to pad it up to 32 bytes. IOW, only
 * two of them can fit in a cacheline in this case. That is OK as it is rare
 * to have more than 2 levels of slowpath nesting in actual use. We don't
 * want to penalize pvqspinlocks to optimize for a rare case in native
 * qspinlocks.
 */
struct qnode {
	struct mcs_spinlock mcs;
#ifdef CONFIG_PARAVIRT_SPINLOCKS
	long reserved[2];
#endif
};

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
 * Exactly fits one 64-byte cacheline on a 64-bit architecture.
 *
 * PV doubles the storage and uses the second cacheline for PV state.
 */
static DEFINE_PER_CPU_ALIGNED(struct qnode, qnodes[MAX_NODES]);

/* Per-CPU pseudo-random number seed */
static DEFINE_PER_CPU(u32, seed);

static inline void set_sleader(struct mcs_spinlock *node, struct mcs_spinlock *qend)
{
	smp_store_release(&node->sleader, 1);
	if (qend != node)
		smp_store_release(&node->last_visited, qend);
}

static inline void clear_sleader(struct mcs_spinlock *node)
{
	node->sleader = 0;
}

static inline void set_waitcount(struct mcs_spinlock *node, int count)
{
	smp_store_release(&node->wcount, count);
}

#define AQS_MAX_LOCK_COUNT      256
#define _Q_LOCKED_PENDING_MASK (_Q_LOCKED_MASK | _Q_PENDING_MASK)

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
#define INTRA_SOCKET_HANDOFF_PROB_ARG	0x10000

static bool probably(void)
{
	u32 v;
	return xor_random() & (INTRA_SOCKET_HANDOFF_PROB_ARG - 1);
	v = this_cpu_read(seed);
	if (v >= 2048) {
		this_cpu_write(seed, 0);
		return false;
	}
	this_cpu_inc(seed);
	return true;
}

#define MAX_POLICY 5
int num_policy = 0;
void *bpf_prog_lock_to_acquire[MAX_POLICY];
void *bpf_prog_lock_acquired[MAX_POLICY];
void *bpf_prog_lock_to_release[MAX_POLICY];
void *bpf_prog_lock_released[MAX_POLICY];

void *bpf_prog_lock_to_enter_slowpath[MAX_POLICY];
void *bpf_prog_lock_enable_fastpath[MAX_POLICY];

void *bpf_prog_should_reorder[MAX_POLICY];
void *bpf_prog_skip_reorder[MAX_POLICY];

void *bpf_prog_lock_bypass_acquire[MAX_POLICY];
void *bpf_prog_lock_bypass_release[MAX_POLICY];

EXPORT_SYMBOL(num_policy);
EXPORT_SYMBOL(bpf_prog_lock_to_acquire);
EXPORT_SYMBOL(bpf_prog_lock_acquired);
EXPORT_SYMBOL(bpf_prog_lock_to_release);
EXPORT_SYMBOL(bpf_prog_lock_released);

EXPORT_SYMBOL(bpf_prog_lock_to_enter_slowpath);
EXPORT_SYMBOL(bpf_prog_lock_enable_fastpath);

EXPORT_SYMBOL(bpf_prog_should_reorder);
EXPORT_SYMBOL(bpf_prog_skip_reorder);

EXPORT_SYMBOL(bpf_prog_lock_bypass_acquire);
EXPORT_SYMBOL(bpf_prog_lock_bypass_release);

// General APIs
static void syncord_lock_to_acquire(struct qspinlock *lock, int policy_id)
{
	return;
}

static void syncord_lock_acquired(struct qspinlock *lock, int policy_id)
{
	return;
}

static void syncord_lock_to_release(struct qspinlock *lock, int policy_id)
{
	return;
}

static void syncord_lock_released(struct qspinlock *lock, int policy_id)
{
	return;
}

// Fastpath APIs
static void syncord_to_enter_slowpath(struct qspinlock *lock, struct mcs_spinlock *node, int policy_id)
{
	return;
}

static bool syncord_enable_fastpath(struct qspinlock *lock, int policy_id)
{
	return true;
}

// Reordering APIs
static int syncord_should_reorder(struct qspinlock *lock, struct mcs_spinlock *node, struct mcs_spinlock *curr, int policy_id)
{
	return (node->nid == curr->nid);
}

static int default_cmp_func(struct qspinlock *lock, struct mcs_spinlock *node, struct mcs_spinlock *curr){
	return (node->nid == curr->nid);
}

static int syncord_skip_reorder(struct qspinlock *lock, struct mcs_spinlock *node){
	return 0;
}

// Lock bypass
static bool syncord_bypass_acquire(struct qspinlock *lock, int policy_id)
{
	return false;
}

static bool syncord_bypass_release(struct qspinlock *lock, int policy_id)
{
	return false;
}


/*
 * This function is responsible for aggregating waiters in a
 * particular socket in one place up to a certain batch count.
 * The invariant is that the shuffle leaders always start from
 * the very next waiter and they are selected ahead in the queue,
 * if required. Moreover, none of the waiters will be behind the
 * shuffle leader, they are always ahead in the queue.
 * Currently, only one shuffle leader is chosen.
 * TODO: Another aggressive approach could be to use HOH locking
 * for n shuffle leaders, in which n corresponds to the number
 * of sockets.
 */
static void shuffle_waiters(struct qspinlock *lock, struct mcs_spinlock *node,
			    int is_next_waiter, int custom, int policy_id)
{
	struct mcs_spinlock *curr, *prev, *next, *last, *sleader, *qend;
	int nid;
	int curr_locked_count;
	int one_shuffle = false;
	int cmp = 0;

	prev = smp_load_acquire(&node->last_visited);
	if (!prev)
		prev = node;
	last = node;
	curr = NULL;
	next = NULL;
	sleader = NULL;
	qend = NULL;

	nid = node->nid;
	curr_locked_count = node->wcount;

	barrier();

	/*
	 * If the wait count is 0, then increase node->wcount
	 * to 1 to avoid coming it again.
	 */
	if (curr_locked_count == 0) {
		set_waitcount(node, ++curr_locked_count);
	}

	/*
         * Our constraint is that we will reset every shuffle
         * leader and the new one will be selected at the end,
         * if any.
         *
         * This one here is to avoid the confusion of having
         * multiple shuffling leaders.
         */
	clear_sleader(node);

	/*
         * In case the curr_locked_count has crossed a
         * threshold, which is certainly impossible in this
         * design, then load the very next of the node and pass
         * the shuffling responsibility to that @next.
         */
	/* if (curr_locked_count >= AQS_MAX_LOCK_COUNT) { */
	if (!probably()) {
		sleader = READ_ONCE(node->next);
		goto out;
	}


	/*
         * In this loop, we try to shuffle the wait queue at
         * least once to allow waiters from the same socket to
         * have no cache-line bouncing. This shuffling is
         * associated in two aspects:
         * 1) when both adjacent nodes belong to the same socket
         * 2) when there is an actual shuffling that happens.
         *
         * Currently, the approach is very conservative. If we
         * miss any of the elements while traversing, we return
         * back.
         *
         * TODO: We can come up with some aggressive strategy to
         * form long chains, which we are yet to explore
         *
         * The way the algorithm works is that it tries to have
         * at least two pointers: pred and curr, in which
         * curr = pred->next. If curr and pred are in the same
         * socket, then no need to shuffle, just update pred to
         * point to curr.
         * If that is not the case, then try to find the curr
         * whose node id is same as the @node's node id. On
         * finding that, we also try to get the @next, which is
         * next = curr->next; which we use all of them to
         * shuffle them wrt @last.
         * @last holds the latest shuffled element in the wait
         * queue, which is updated on each shuffle and is most
         * likely going to be next shuffle leader.
         */
	for (;;) {
		/*
		 * Get the curr first
		 */
		curr = READ_ONCE(prev->next);

		/*
                 * Now, right away we can quit the loop if curr
                 * is NULL or is at the end of the wait queue
                 * and choose @last as the sleader.
                 */
		if (!curr) {
			sleader = last;
			qend = prev;
			break;
		}

	     recheck_curr_tail:
                /*
                 * If we are the last one in the tail, then
                 * we cannot do anything, we should return back
                 * while selecting the next sleader as the last one
                 */
		if (curr->cid == (atomic_read(&lock->val) >> _Q_TAIL_CPU_OFFSET)) {
			sleader = last;
			qend = prev;
			break;
		}

		/* got the current for sure */
		if(custom)
			cmp = syncord_should_reorder(lock, node, curr, policy_id);
		else
			cmp = default_cmp_func(lock, node, curr);
		/* Check if curr->nid is same as nid */
		if (cmp) {

			/*
			 * if prev->nid == curr->nid, then
			 * just update the last and prev
			 * and proceed forward
			 */
			if (prev == last) {
				set_waitcount(curr, curr_locked_count);

				last = curr;
				prev = curr;
				one_shuffle = true;

			} else {
				/* prev->nid is not same, then we need
				 * to find next and move @curr to
				 * last->next, while linking @prev->next
				 * to next.
				 *
				 * NOTE: We do not update @prev here
				 * because @curr has been already moved
				 * out.
				 */
				next = READ_ONCE(curr->next);
				if (!next) {
					sleader = last;
					qend = prev;
					/* qend = curr; */
					break;
				}

				/*
                                 * Since, we have curr and next,
                                 * we mark the curr that it has been
                                 * shuffled and shuffle the queue
                                 */
				set_waitcount(curr, curr_locked_count);

/*
 *                                                 (1)
 *                                    (3)       ----------
 *                          -------------------|--\      |
 *                        /                    |   v     v
 *   ----          ----   |  ----        ----/   ----   ----
 *  | SL | -> ... |Last| -> | X  |....->|Prev|->|Curr|->|Next|->....
 *   ----          ----  ->  ----        ----    ----  | ----
 *                      /          (2)                /
 *                      -----------------------------
 *                              |
 *                              V
 *   ----          ----      ----      ----        ----    ----
 *  | SL | -> ... |Last| -> |Curr| -> | X  |....->|Prev|->|Next|->....
 *   ----          ----      ----      ----        ----    ----
 *
 */
				prev->next = next;
				curr->next = last->next;
				last->next = curr;
				smp_wmb();

				last = curr;
				curr = next;
				one_shuffle = true;

				goto recheck_curr_tail;
			}
		} else
			prev = curr;

		/*
		 * Currently, we only exit once we have at least
		 * one shuffler if the shuffling leader is the
		 * very next lock waiter.
		 * TODO: This approach can be further optimized.
		 */
		if (one_shuffle) {
			if ((is_next_waiter &&
			     !(atomic_read_acquire(&lock->val) & _Q_LOCKED_PENDING_MASK)) ||
			    (!is_next_waiter && READ_ONCE(node->lstatus))) {
				sleader = last;
				qend = prev;
				break;
			}
		}
	}

     out:
	if (sleader) {
		set_sleader(sleader, qend);
	}
}

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

	return per_cpu_ptr(&qnodes[idx].mcs, cpu);
}

static inline __pure
struct mcs_spinlock *grab_mcs_node(struct mcs_spinlock *base, int idx)
{
	return &((struct qnode *)base + idx)->mcs;
}

/* #define _Q_LOCKED_PENDING_MASK (_Q_LOCKED_MASK | _Q_PENDING_MASK) */

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
 * queued_fetch_set_pending_acquire - fetch the whole lock value and set pending
 * @lock : Pointer to queued spinlock structure
 * Return: The previous lock value
 *
 * *,*,* -> *,1,*
 */
#ifndef queued_fetch_set_pending_acquire
static __always_inline u32 queued_fetch_set_pending_acquire(struct qspinlock *lock)
{
	return atomic_fetch_or_acquire(_Q_PENDING_VAL, &lock->val);
}
#endif

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
void queued_spin_lock_slowpath(struct qspinlock *lock, u32 val, int custom, int policy_id)
{
	struct mcs_spinlock *prev, *next, *node;
	u32 old, tail;
	int idx;
	int cid;

	BUILD_BUG_ON(CONFIG_NR_CPUS >= (1U << _Q_TAIL_CPU_BITS));

	/* if (pv_enabled()) */
	/* 	goto pv_queue; */

	/* if (virt_spin_lock(lock)) */
	/* 	return; */

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
	 * 0,0,* -> 0,1,* -> 0,0,1 pending, trylock
	 */
	val = queued_fetch_set_pending_acquire(lock);

	/*
	 * If we observe contention, there is a concurrent locker.
	 *
	 * Undo and queue; our setting of PENDING might have made the
	 * n,0,0 -> 0,0,0 transition fail and it will now be waiting
	 * on @next to become !NULL.
	 */
	if (unlikely(val & ~_Q_LOCKED_MASK)) {

		/* Undo PENDING if we set it. */
		if (!(val & _Q_PENDING_MASK))
			clear_pending(lock);

		goto queue;
	}

	/*
	 * We're pending, wait for the owner to go away.
	 *
	 * 0,1,1 -> 0,1,0
	 *
	 * this wait loop must be a load-acquire such that we match the
	 * store-release that clears the locked bit and create lock
	 * sequentiality; this is because not all
	 * clear_pending_set_locked() implementations imply full
	 * barriers.
	 */
	if (val & _Q_LOCKED_MASK)
		atomic_cond_read_acquire(&lock->val, !(VAL & _Q_LOCKED_MASK));

	/*
	 * take ownership and clear the pending bit.
	 *
	 * 0,1,0 -> 0,0,1
	 */
	clear_pending_set_locked(lock);
	lockevent_inc(lock_pending);
	return;

	/*
	 * End of pending bit optimistic spinning and beginning of MCS
	 * queuing.
	 */
queue:
	lockevent_inc(lock_slowpath);
pv_queue:
	node = this_cpu_ptr(&qnodes[0].mcs);
	idx = node->count++;
	cid = smp_processor_id();
	tail = encode_tail(cid, idx);

	/*
	 * 4 nodes are allocated based on the assumption that there will
	 * not be nested NMIs taking spinlocks. That may not be true in
	 * some architectures even though the chance of needing more than
	 * 4 nodes will still be extremely unlikely. When that happens,
	 * we fall back to spinning on the lock directly without using
	 * any MCS node. This is not the most elegant solution, but is
	 * simple enough.
	 */
	if (unlikely(idx >= MAX_NODES)) {
		lockevent_inc(lock_no_node);
		while (!queued_spin_trylock(lock))
			cpu_relax();
		goto release;
	}

	node = grab_mcs_node(node, idx);

	/*
	 * Keep counts of non-zero index values:
	 */
	lockevent_cond_inc(lock_use_node2 + idx - 1, idx);

	/*
	 * Ensure that we increment the head node->count before initialising
	 * the actual node. If the compiler is kind enough to reorder these
	 * stores, then an IRQ could overwrite our assignments.
	 */
	barrier();

	node->cid = cid +1;
	node->nid = numa_node_id();
	node->last_visited = NULL;
	node->locked = 0;
	node->next = NULL;
	node->bpf_args = NULL;
	pv_init_node(node);

	if(custom){
		syncord_to_enter_slowpath(lock, node, policy_id);
	}
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

		/* Link @node into the waitqueue. */
		WRITE_ONCE(prev->next, node);

		pv_wait_node(node, prev);
		/** arch_mcs_spin_lock_contended(&node->locked); */

		for (;;) {
			int __val = READ_ONCE(node->lstatus);
			if (__val)
				break;

			if (READ_ONCE(node->sleader))
				shuffle_waiters(lock, node, false, custom, policy_id);

			cpu_relax();
		}
		smp_acquire__after_ctrl_dep();

		/*
		 * While waiting for the MCS lock, the next pointer may have
		 * been set by another lock waiter. We optimistically load
		 * the next pointer & prefetch the cacheline for writing
		 * to reduce latency in the upcoming MCS unlock operation.
		 */
		/* next = READ_ONCE(node->next); */
		/* if (next) */
		/* 	prefetchw(next); */
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
	/** if ((val = pv_wait_head_or_lock(lock, node))) */
	/**     goto locked; */

	/** val = atomic_cond_read_acquire(&lock->val, !(VAL & _Q_LOCKED_PENDING_MASK)); */
	for (;;) {
		int wcount;

		val = atomic_read(&lock->val);
		if (!(val & _Q_LOCKED_PENDING_MASK))
			break;

		wcount = READ_ONCE(node->wcount);
		if (!wcount ||
		    (wcount && node->sleader))
			shuffle_waiters(lock, node, true, custom, policy_id);
		cpu_relax();
	}
	smp_acquire__after_ctrl_dep();

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
	 * In the PV case we might already have _Q_LOCKED_VAL set, because
	 * of lock stealing; therefore we must also allow:
	 *
	 * n,0,1 -> 0,0,1
	 *
	 * Note: at this point: (val & _Q_PENDING_MASK) == 0, because of the
	 *       above wait condition, therefore any concurrent setting of
	 *       PENDING will make the uncontended transition fail.
	 */
	if ((val & _Q_TAIL_MASK) == tail) {
		if (atomic_try_cmpxchg_relaxed(&lock->val, &val, _Q_LOCKED_VAL))
			goto release; /* No contention */
	}

	/*
	 * Either somebody is queued behind us or _Q_PENDING_VAL got set
	 * which will then detect the remaining tail and queue behind us
	 * ensuring we'll see a @next.
	 */
	set_locked(lock);

	/*
	 * contended path; wait for next if not observed yet, release.
	 */
	next = smp_load_acquire(&node->next);
	if (!next)
		next = smp_cond_load_relaxed(&node->next, (VAL));

	/* arch_mcs_spin_unlock_contended(&next->locked); */
	smp_store_release(&next->lstatus, 1);
	pv_kick_node(lock, next);

release:
	/*
	 * release the node
	 */
	__this_cpu_dec(qnodes[0].mcs.count);
}
EXPORT_SYMBOL(queued_spin_lock_slowpath);


void bpf_queued_spin_lock(struct qspinlock *lock, int policy_id)
{
	u32 val = 0;
	syncord_lock_to_acquire(lock, policy_id);

	if(unlikely(syncord_bypass_acquire(lock, policy_id))){
		return;
	}

	if (likely(syncord_enable_fastpath(lock, policy_id)) &&
			likely(atomic_try_cmpxchg_acquire(&lock->val, &val, _Q_LOCKED_VAL))){
		syncord_lock_acquired(lock, policy_id);
		return;
	}

	queued_spin_lock_slowpath(lock, val, 1, policy_id);
	syncord_lock_acquired(lock, policy_id);
}
EXPORT_SYMBOL(bpf_queued_spin_lock);

void bpf_queued_spin_unlock(struct qspinlock *lock, int policy_id)
{
	syncord_lock_to_release(lock, policy_id);

	if(unlikely(syncord_bypass_release(lock, policy_id))){
		return;
	}

	smp_store_release(&lock->locked, 0);
	syncord_lock_released(lock, policy_id);
}
EXPORT_SYMBOL(bpf_queued_spin_unlock);

/**
 * queued_spin_lock - acquire a queued spinlock
 * @lock: Pointer to queued spinlock structure
 */
void queued_spin_lock(struct qspinlock *lock)
{
	u32 val = 0;

	if (likely(atomic_try_cmpxchg_acquire(&lock->val, &val, _Q_LOCKED_VAL)))
		return;

	queued_spin_lock_slowpath(lock, val, 0, 0);
}
EXPORT_SYMBOL(queued_spin_lock);

/**
 * queued_spin_unlock - release a queued spinlock
 * @lock : Pointer to queued spinlock structure
 */
void queued_spin_unlock(struct qspinlock *lock)
{
	/*
	 * unlock() needs release semantics:
	 */
	smp_store_release(&lock->locked, 0);
}
EXPORT_SYMBOL(queued_spin_unlock);

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
