package utilities.iteration;

import utilities.ArrayUtilities;
import utilities.StrUtils;
import weka.core.OptionHandler;
import weka.core.Randomizable;

import java.io.Serializable;
import java.util.*;

public class RandomIterator<A>
    implements DefaultListIterator<A>,
               OptionHandler,
               Randomizable,
               Serializable {

    public List<A> getList() {
        return list;
    }

    public void setList(final List<A> list) {
        this.list = list;
        indices = ArrayUtilities.sequence(list.size()); // todo remove dependency on indices list
    }

    protected Random random = null;
    protected Integer seed = null;
    protected List<A> list = new ArrayList<>();
    protected List<Integer> indices;
    protected int index = -1;
    public final static String SEED_FLAG = "-s";
    protected boolean nextIndexSetup = false;

    public boolean isRemovedOnNext() {
        return removedOnNext;
    }

    public void setRemovedOnNext(final boolean removedOnNext) {
        this.removedOnNext = removedOnNext;
    }

    protected boolean removedOnNext = true;

    protected void setRandomIndex() {
        if(!nextIndexSetup) {
            if(random == null) {
                if(seed == null) {
                    throw new IllegalStateException("seed not set");
                }
                random = new Random(seed);
            }
            seed = random.nextInt();
            if(indices.isEmpty()) {
                index = -1;
            } else {
                index = random.nextInt(indices.size());
            }
            random.setSeed(seed);
            nextIndexSetup = true;
        }
    }

    public RandomIterator() {
    }

    public RandomIterator(int seed) {
        setSeed(seed);
    }

    public RandomIterator(int seed, List<A> list) {
        setSeed(seed);
        setList(list);
    }

    public RandomIterator(List<A> list) {
        this(-1, list);
    }

    public RandomIterator(List<A> list, int seed) {
        this(seed, list);
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    @Override
    public int nextIndex() {
        setRandomIndex();
        return index;
    }

    @Override
    public A next() {
        int index = nextIndex();
        if(removedOnNext) {
            index = indices.remove(index);
        } else {
            index = indices.get(index);
        }
        A element = list.get(index);
        nextIndexSetup = false;
        return element;
    }

    @Override
    public boolean hasNext() {
        return !indices.isEmpty();
    }

    @Override
    public void add(final A item) {
        list.add(item);
        indices.add(list.size() - 1);
    }

    @Override
    public void remove() {
        if(!removedOnNext) {
            indices.remove(index);
        }
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        StrUtils.setOption(options, SEED_FLAG, this::setSeed, Integer::parseInt);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        options.add(SEED_FLAG); options.add(String.valueOf(seed));
        return options.toArray(new String[0]);
    }
}
