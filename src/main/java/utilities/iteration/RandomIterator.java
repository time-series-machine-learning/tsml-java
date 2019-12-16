package utilities.iteration;

import utilities.StringUtilities;
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
    }

    private Random random = null;
    private Integer seed = null;
    private List<A> list = new ArrayList<>(); // todo tree list?
    private int index;
    public final static String SEED_FLAG = "-s";
    private boolean nextIndexSetup = false;

    private void setRandomIndex() {
        if(!nextIndexSetup) {
            if(random == null) {
                if(seed == null) {
                    throw new IllegalStateException("seed not set");
                }
                random = new Random(seed);
            }
            seed = random.nextInt();
            if(list.isEmpty()) {
                index = 0;
            } else {
                index = random.nextInt(list.size());
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
        A element = list.get(nextIndex());
        nextIndexSetup = false;
        return element;
    }

    @Override
    public boolean hasNext() {
        return !list.isEmpty();
    }

    @Override
    public void add(final A item) {
        list.add(item);
    }

    @Override
    public void remove() {
        list.remove(index);
        nextIndexSetup = false;
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        StringUtilities.setOption(options, SEED_FLAG, this::setSeed, Integer::parseInt);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        options.add(SEED_FLAG); options.add(String.valueOf(seed));
        return options.toArray(new String[0]);
    }
}
