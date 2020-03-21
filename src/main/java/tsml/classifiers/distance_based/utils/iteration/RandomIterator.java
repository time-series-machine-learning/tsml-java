package tsml.classifiers.distance_based.utils.iteration;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.random.BaseRandom;
import utilities.ArrayUtilities;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class RandomIterator<A> extends BaseRandom implements RandomIteration<A> {

    private List<A> list;
    private List<Integer> indices;
    private int index = -1;
    private boolean nextIndexSetup = false;
    protected boolean replacement = true;

    protected boolean isNextIndexSetup() {
        return nextIndexSetup;
    }

    protected RandomIterator<A> setNextIndexSetup(final boolean nextIndexSetup) {
        this.nextIndexSetup = nextIndexSetup;
        return this;
    }

    protected List<Integer> getIndices() {
        return indices;
    }

    protected RandomIterator<A> setIndices(final List<Integer> indices) {
        this.indices = indices;
        return this;
    }

    protected int getIndex() {
        return index;
    }

    protected RandomIterator<A> setIndex(final int index) {
        this.index = index;
        return this;
    }

    public boolean withReplacement() {
        return replacement;
    }

    public RandomIterator<A> setReplacement(final boolean replacement) {
        this.replacement = replacement;
        return this;
    }

    public List<A> getList() {
        return list;
    }

    public void setList(final List<A> list) {
        Assert.assertNotNull(list);
        this.list = list;
        setIndices(ArrayUtilities.sequence(list.size()));
    }


    protected void setRandomIndex() {
        if(!isNextIndexSetup()) {
            if(getIndices().isEmpty()) {
                setIndex(-1);
            } else {
                setIndex(getRandom().nextInt(getIndices().size()));
            }
            setNextIndexSetup(true);
        }
    }

    public RandomIterator(int seed) {
        this(seed, new ArrayList<>());
    }

    public RandomIterator(Random random) {
        this(random, new ArrayList<>());
    }

    public RandomIterator(int seed, List<A> list) {
        super(seed);
        setList(list);
    }

    public RandomIterator(Random random, List<A> list) {
        super(random);
        setList(list);
    }

    public int nextIndex() {
        setRandomIndex();
        return getIndex();
    }

    public A next() {
        int index = nextIndex();
        if(withReplacement()) {
            index = getIndices().remove(index);
        } else {
            index = getIndices().get(index);
        }
        A element = getList().get(index);
        setNextIndexSetup(false);
        return element;
    }

    @Override
    public boolean hasNext() {
        return !getIndices().isEmpty();
    }

    public void remove() {
        if(!withReplacement()) {
            getIndices().remove(getIndex());
        }
    }

    // todo checks on multiple next / remove calls
}
