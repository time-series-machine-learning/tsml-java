package tsml.classifiers.distance_based.utils.collections.iteration;

import tsml.classifiers.distance_based.utils.collections.CollectionUtils;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import static utilities.ArrayUtilities.sequence;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class BaseRandomIterator<A> extends AbstractListIterator<A> implements RandomIterator<A> {

    private boolean withReplacement = true;
    private boolean skipSingleOption = true;
    private boolean allowReordering = true;
    private List<Integer> indices;
    private int indexOfIndex = -1;
    private Random random;

    @Override public void add(final A a) {
        indices.add(getList().size());
        super.add(a);
    }

    @Override public boolean withReplacement() {
        return withReplacement;
    }

    @Override public void setWithReplacement(final boolean withReplacement) {
        this.withReplacement = withReplacement;
    }

    @Override public void buildIterator(final List<A> list) {
        super.buildIterator(list);
        indices = sequence(list.size());
        Objects.requireNonNull(random);
    }

    @Override protected int findNextIndex() {
        final int size = indices.size();
        if(skipSingleOption && size == 1) {
            indexOfIndex = 0;
        } else {
            indexOfIndex = random.nextInt(size);
        }
        return indices.get(indexOfIndex);
    }

    @Override public A next() {
        final A element = super.next();
        // if not choosing with replacement then remove the index from the pool of indices to be picked from next time
        if(!withReplacement) {
            removeIndex();
        }
        return element;
    }

    @Override
    protected boolean findHasNext() {
        return !indices.isEmpty();
    }

    private void removeIndex() {
        CollectionUtils.remove(indices, indexOfIndex, allowReordering);
    }
    
    @Override public void remove() {
        super.remove();
        // if choosing with replacement then the index has been kept during the next() call. The remove() call signals that it should be removed.
        if(withReplacement) {
            removeIndex();
        }
    }

    @Override public void setRandom(final Random random) {
        this.random = random;
    }

    @Override public Random getRandom() {
        return random;
    }

    @Override public boolean isSkipSingleOption() {
        return skipSingleOption;
    }

    @Override public void setSkipSingleOption(final boolean skipSingleOption) {
        this.skipSingleOption = skipSingleOption;
    }

    public boolean isAllowReordering() {
        return allowReordering;
    }

    public void setAllowReordering(final boolean allowReordering) {
        this.allowReordering = allowReordering;
    }
}
