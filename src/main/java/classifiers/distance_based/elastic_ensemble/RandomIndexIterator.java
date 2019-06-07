package classifiers.distance_based.elastic_ensemble;

import utilities.ArrayUtilities;

import java.util.*;

public class RandomIndexIterator implements Iterator<Integer> {

    private final Random random;
    private final int size;
    private final List<Integer> indices;

    public RandomIndexIterator(final Random random, final int size) {this.random = random;
        this.size = size;
        indices = new ArrayList<>(Arrays.asList(ArrayUtilities.box(ArrayUtilities.range(size))));
    }

    @Override
    public boolean hasNext() {
        return !indices.isEmpty();
    }

    @Override
    public Integer next() {
        return indices.remove(random.nextInt(indices.size()));
    }
}
