package tsml.classifiers.distance_based.utils.collections.views;

import tsml.classifiers.distance_based.utils.collections.DefaultList;

public class IntListView implements DefaultList<Integer> {

    final int[] array;

    public IntListView(final int[] array) {this.array = array;}

    @Override public Integer get(final int i) {
        return array[i];
    }

    @Override public Integer set(final int i, final Integer integer) {
        int prev = array[i];
        array[i] = integer;
        return prev;
    }

    @Override public int size() {
        return array.length;
    }
}
