package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RoundRobinIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class RoundRobinRandomSampler
    extends AbstractIterator<Instance> {

    private final RoundRobinIterator<AbstractIterator<Instance>> samplerIterator;
    private AbstractIterator<Instance> sampler;
    private final Map<Double, AbstractIterator<Instance>> samplers = new HashMap<>();
    private final Random random = new Random();

    public RoundRobinRandomSampler(final Collection<Instance> instances, final Random random) { // todo update to seeding system
        this(instances, random.nextLong());
    }

    public RoundRobinRandomSampler(final Collection<Instance> instances, final long seed) {
        this(seed);
        for(Instance instance : instances) {
            add(instance);
        }
    }

    public RoundRobinRandomSampler(long seed) {
        this.random.setSeed(seed);
        samplerIterator = new RoundRobinIterator<>();
    }

    public RoundRobinRandomSampler(Random random) {
        this(random.nextLong());
    }

    public RoundRobinRandomSampler(RoundRobinRandomSampler other) {
        random.setSeed(other.random.nextLong());
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
            AbstractIterator<Instance> sampler = new RandomSampler(random.nextLong());
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
