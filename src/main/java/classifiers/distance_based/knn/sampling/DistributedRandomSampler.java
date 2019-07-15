package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import utilities.CollectionUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class DistributedRandomSampler
    extends AbstractIterator<Instance> {
    private final Random random = new Random();
    private final Map<Double, List<Instance>> instancesByClass;
    private final List<ClassLikelihood> probabilities = new ArrayList<>();
    private List<Instance> instances;
    private int index = 0;
    private double classValue;

    @Override
    public DistributedRandomSampler iterator() {
        return new DistributedRandomSampler(this);
    }

    private static class ClassLikelihood
        implements Comparable<ClassLikelihood> {
        private final double classValue;

        public double getClassValue() {
            return classValue;
        }

        private double probability;
        private final double classDistributionProbability;

        private ClassLikelihood(final double classValue,
                                final double classDistributionProbability) {
            this.classValue = classValue;
            this.probability = classDistributionProbability;
            this.classDistributionProbability = classDistributionProbability;
        }

        private ClassLikelihood(ClassLikelihood other) {
            this(other.classValue, other.classDistributionProbability);
            this.probability = other.probability;
        }

        public void chosen() {
            probability--;
        }

        public void increment() {
            probability += classDistributionProbability;
        }

        @Override
        public int compareTo(final ClassLikelihood classLikelihood) {
            return Double.compare(probability, classLikelihood.probability);
        }
    }

    public DistributedRandomSampler(long seed, Collection<Instance> instances) {
        random.setSeed(seed);
        instancesByClass = Utilities.instancesByClassValue(instances);
        for(Map.Entry<Double, List<Instance>> entry : instancesByClass.entrySet()) {
            double probability = (double) entry.getValue().size() / instances.size();
            probabilities.add(new ClassLikelihood(entry.getKey(), probability));
        }
    }

    public DistributedRandomSampler(long seed) {
        this(seed, new ArrayList<>());
    }

    public DistributedRandomSampler(Random random) {
        this(random.nextLong());
    }

    public DistributedRandomSampler(DistributedRandomSampler other) {
        this(other.random);
        index = other.index;
        for(Map.Entry<Double, List<Instance>> entry : other.instancesByClass.entrySet()) {
            instancesByClass.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        for(ClassLikelihood classLikelihood : other.probabilities) {
            probabilities.add(new ClassLikelihood(classLikelihood));
        }
        classValue = other.classValue;
        instances = instancesByClass.get(classValue);
    }

    @Override
    public boolean hasNext() {
        return !instancesByClass.isEmpty();
    }

    @Override
    public Instance next() {
        ClassLikelihood classLikelihood = CollectionUtilities.findBest(probabilities, random);
        classLikelihood.chosen();
        for(ClassLikelihood a : probabilities) {
            a.increment();
        }
        classValue = classLikelihood.getClassValue();
        instances = instancesByClass.get(classValue);
        index = random.nextInt(instances.size());
        return instances.get(index);
    }

    @Override
    public void remove() {
        instances.remove(index);
    }

    @Override
    public void add(final Instance instance) {
        throw new UnsupportedOperationException(); // todo
    }


}
