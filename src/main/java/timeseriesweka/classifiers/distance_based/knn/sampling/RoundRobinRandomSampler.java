package timeseriesweka.classifiers.distance_based.knn.sampling;

import utilities.iteration.AbstractIterator;
import utilities.iteration.linear.RoundRobinIterator;
import utilities.iteration.random.RandomIterator;
import utilities.samplers.RandomSampler;
import weka.core.Instance;

import java.util.*;

public class RoundRobinRandomSampler
    extends AbstractIterator<Instance> {

    private final RoundRobinIterator<AbstractIterator<Instance>> samplerIterator;
    private AbstractIterator<Instance> sampler;
    private final Map<Double, AbstractIterator<Instance>> samplers = new HashMap<>();
    private final Random random = new Random();
    private long seed;

    public RoundRobinRandomSampler(final Random random, final Collection<Instance> instances) { // todo update to seeding system
        this(random.nextLong(), instances);
    }

    public RoundRobinRandomSampler(final long seed, final Collection<Instance> instances) {
        this.seed = seed;
        random.setSeed(seed);
        samplerIterator = new RoundRobinIterator<>();
        for(Instance instance : instances) {
            add(instance);
        }
    }

    public RoundRobinRandomSampler(long seed) {
        this(seed, new ArrayList<>());
    }

    public RoundRobinRandomSampler(Random random) {
        this(random, new ArrayList<>());
    }

    public RoundRobinRandomSampler(RoundRobinRandomSampler other) {
        random.setSeed(other.seed);
        seed = other.seed;
        // todo make sure the copying of random iterators results in same pattern, or does this not matter as randomness is still random?
        sampler = other.sampler.iterator();
        for(Map.Entry<Double, AbstractIterator<Instance>> entry : other.samplers.entrySet()) {
            samplers.put(entry.getKey(), entry.getValue().iterator());
        }
        samplerIterator = other.samplerIterator.iterator();
    }

    @Override
    public void remove() {
        sampler.remove();
        if(!sampler.hasNext()) {
            samplerIterator.remove();
        }
    }

    @Override
    public void add(final Instance instance) {
        double classValue = instance.classValue();
        AbstractIterator<Instance> instanceIterator = samplers.computeIfAbsent(classValue, key -> {
            AbstractIterator<Instance> sampler = new RandomIterator<>(random);
            samplerIterator.add(sampler);
            return sampler;
        });
        instanceIterator.add(instance);
    }

    @Override
    public boolean hasNext() {
        return samplerIterator.hasNext();
    }

    @Override
    public Instance next() {
        sampler = samplerIterator.next();
        return sampler.next();
    }

    @Override
    public RoundRobinRandomSampler iterator() {
        return new RoundRobinRandomSampler(this);
    }
}
