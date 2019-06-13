package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.RoundRobinIterator;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RoundRobinRandomSampler implements Sampler {

    private final RoundRobinIterator<RandomSampler> randomSamplerIterator;
    private RandomSampler randomSampler;

    public RoundRobinRandomSampler(final Instances instances, final Random random) {
        List<RandomSampler> randomSamplers = new ArrayList<>();
        Map<Double, List<Instance>> classMap = Utilities.instancesByClassValue(instances);
        for(Map.Entry<Double, List<Instance>> entry : classMap.entrySet()) {
            RandomSampler randomSampler = new RandomSampler(entry.getValue(), random);
            if(randomSampler.hasNext()) {
                randomSamplers.add(randomSampler);
            }
        }
        randomSamplerIterator = new RoundRobinIterator<>(randomSamplers);
    }

    @Override
    public void remove() {
        randomSampler.remove();
        if(!randomSampler.hasNext()) {
            randomSamplerIterator.remove();
        }
    }

    @Override
    public boolean hasNext() {
        return randomSamplerIterator.hasNext();
    }

    @Override
    public Instance next() {
        randomSampler = randomSamplerIterator.next();
        Instance instance = randomSampler.next();

        return instance;
    }
}
