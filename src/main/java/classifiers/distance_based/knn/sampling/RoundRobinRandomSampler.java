package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RoundRobinIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class RoundRobinRandomSampler
    extends DynamicIterator<Instance, RoundRobinRandomSampler> {

    private final RoundRobinIterator<ListIterator<Instance>> samplerIterator = new RoundRobinIterator<>();
    private ListIterator<Instance> sampler;
    private final Map<Double, ListIterator<Instance>> samplers = new HashMap<>();
    private final Random random;

    public RoundRobinRandomSampler(final Instances instances, final Random random) { // todo update to seeding system
        throw new UnsupportedOperationException();
    }

    public RoundRobinRandomSampler(Random random) {
        this.random = random;
    }

    public RoundRobinRandomSampler(RoundRobinRandomSampler other) {
        this(other.random);
        throw new UnsupportedOperationException();
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
        ListIterator<Instance> instanceIterator = samplers.computeIfAbsent(classValue, key -> {
            RandomSampler sampler = new RandomSampler(random.nextLong());
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
