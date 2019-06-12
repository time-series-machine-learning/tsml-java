package classifiers.distance_based.knn.sampling;

import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.BiFunction;

public class DistributedRandomSampler implements Sampler {
    private final Random random;
    private final Map<Double, Instances> instancesByClass;
    private final TreeMap<Double, Double> classDistribution = new TreeMap<>();
    private final TreeMap<Double, Double> classProbabilities = new TreeMap<>();

    public DistributedRandomSampler(Instances instances, Random random) {
        this.random = random;
        instancesByClass = Utilities.instancesByClassValue(instances);
        for(Map.Entry<Double, Instances> entry : instancesByClass.entrySet()) {
            double probability = (double) entry.getValue().size() / instances.size();
            classDistribution.put(entry.getKey(), probability);
            classProbabilities.put(entry.getKey(), probability);
        }
    }

    @Override
    public boolean hasNext() {
        return !instancesByClass.isEmpty();
    }

    @Override
    public Instance next() {
        Map.Entry<Double, Double> lastEntry = classProbabilities.pollLastEntry();
        double classValue = lastEntry.getValue();
        double classProbability = lastEntry.getKey();
        for(Map.Entry<Double, Double> entry : classProbabilities.entrySet()) {
            classDistribution.compute(entry.getKey(), (key, currentProbability) -> currentProbability + classProbability);
        }
        classDistribution.compute(classValue, (key, currentProbability) -> currentProbability - 1);

        
    }

    @Override
    public void remove() {

    }
}
