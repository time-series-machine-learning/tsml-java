package utilities.collections;

import com.sun.istack.internal.NotNull;

public class IntListView implements DefaultList<Integer> {

    @NotNull final int[] array;

    public IntListView(@NotNull final int[] array) {this.array = array;}

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
