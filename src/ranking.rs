use std::cmp::Ordering;

/// Descending comparator for similarity scores that defines a true total order:
/// bare `partial_cmp(...).unwrap_or(Equal)` treats every NaN as tied with every
/// finite score, which breaks `sort_by`'s ordering contract and can leave NaN
/// hits scattered through (or even ahead of) valid results. NaN is explicitly
/// the worst score and always sinks to the end, regardless of its sign bit;
/// finite scores use `total_cmp` for a well-defined order among themselves.
pub(crate) fn score_cmp_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // a is NaN: a sorts after (worse than) b
        (false, true) => Ordering::Less,    // b is NaN: b sorts after (worse than) a
        (false, false) => b.total_cmp(&a),  // descending among finite scores
    }
}

/// Keep the best `k` items under `cmp` (which must be a total order), sorted.
///
/// Partial selection first, so a large index with a small `k` does not pay for a
/// full sort.
pub(crate) fn top_k<T, F>(items: &mut Vec<T>, k: usize, mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if k == 0 {
        items.clear();
        return;
    }
    if items.len() > k {
        items.select_nth_unstable_by(k - 1, &mut cmp);
        items.truncate(k);
    }
    items.sort_by(cmp);
}
