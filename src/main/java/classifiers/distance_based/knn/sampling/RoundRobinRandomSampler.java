package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.AbstractRoundRobinIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RoundRobinIterator;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class RoundRobinRandomSampler
    extends DynamicIterator<Instance, RoundRobinRandomSampler> {

    private RoundRobinIterator<ListIterator<Instance>> randomSamplerIterator;
    private ListIterator<Instance> randomSampler;
    private final Map<Double, ListIterator<Instance>> randomSamplers = new HashMap<>();
    private final Random random;

    public RoundRobinRandomSampler(final Instances instances, final Random random) {
        Map<Double, List<Instance>> classMap = Utilities.instancesByClassValue(instances);
        for(Map.Entry<Double, List<Instance>> entry : classMap.entrySet()) {
            RandomSampler randomSampler = new RandomSampler(entry.getValue(), random);
            if(randomSampler.hasNext()) {
                randomSamplers.put(entry.getKey(), randomSampler);
            }
        }
        this.random = random;
        randomSamplerIterator = new RoundRobinIterator<>(randomSamplers.values());
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
        randomSampler.remove();
        if(!randomSampler.hasNext()) {
            randomSamplerIterator.remove();
        }
    }

    @Override
    public void add(final Instance instance) {
        double classValue = instance.classValue();
        randomSamplers.get(classValue).add(instance);
    }

    @Override
    public boolean hasNext() {
        return randomSamplerIterator.hasNext();
    }

    @Override
    public Instance next() {
        randomSampler = randomSamplerIterator.next();
        return randomSampler.next();
    }

    @Override
    public RoundRobinRandomSampler iterator() {
        return new RoundRobinRandomSampler(this);
    }
}
