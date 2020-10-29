package tsml.classifiers.distance_based.utils.collections.iteration;

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
    private List<Integer> indices;
    private int indexOfIndex;
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
        if(size == 1) {
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
    public boolean hasNext() {
        return !indices.isEmpty();
    }

    private void removeIndex() {
        // rather than removing the element at indexOfIndex and shifting all elements after indexOfIndex down 1, it is more efficient to swap the last element in place of the element being removed and remove the last element. I.e. [1,2,3,4,5] and indexOfIndex=2. Swap in the end element, [1,2,5,4,5], and remove the last element, [1,2,5,4].
        final int lastIndex = indices.size() - 1;
        final Integer lastElement = indices.get(lastIndex);
        indices.remove(lastIndex);
        // if the indexOfIndex is the last element then don't bother swapping, removal is enough.
        if(indexOfIndex != lastIndex) {
            // otherwise need to set the end element in place of the current
            indices.set(indexOfIndex, lastElement);
        }
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
    
}
