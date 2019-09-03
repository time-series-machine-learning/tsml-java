package timeseriesweka.classifiers.distance_based.knn.sampling;

import utilities.ArrayUtilities;
import utilities.iteration.AbstractIterator;
import utilities.iteration.random.RandomIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class DistributedRandomSampler
    extends AbstractIterator<Instance> {

    private Random random = new Random();
    private Long seed;
    private Sampler previous;
    private final List<Sampler> samplers = new ArrayList<>();
    private final Map<Double, Sampler> map = new HashMap<>();

    @Override
    public DistributedRandomSampler iterator() {
        throw new UnsupportedOperationException();
//        return new DistributedRandomSampler(this);
    }

    private static class Sampler extends ClassLikelihood {

        private final RandomIterator<Instance> iterator = new RandomIterator<>();

        private Sampler(final double classValue, final double classDistributionProbability) {
            super(classValue, classDistributionProbability);
        }

        public RandomIterator<Instance> getIterator() {
            return iterator;
        }

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

    public void setDistribution(Instances instances) {
        samplers.clear();
        map.clear();
        int[] distribution = new int[instances.numClasses()];
        for(Instance instance : instances) {
            distribution[(int) instance.classValue()]++;
        }
        for (int i = 0; i < instances.numClasses(); i++) {
            Sampler sampler = new Sampler(i, distribution[i]);
            samplers.add(i, sampler);
            map.put((double) i, sampler);
        }
    }

    public DistributedRandomSampler(long seed) {
        this.seed = seed;
        random.setSeed(seed);
    }

    public DistributedRandomSampler(Random random) {
        this.random = random; // todo sort out
    }

    @Override
    public boolean hasNext() {
        return !samplers.isEmpty();
    }

    @Override
    public Instance next() {
        Sampler sampler = ArrayUtilities.best(samplers, random);
        sampler.chosen();
        for(ClassLikelihood a : samplers) {
            a.increment();
        }
        RandomIterator<Instance> iterator = sampler.getIterator();
        previous = sampler;
        return iterator.next();
    }

    @Override
    public void remove() {
        previous.getIterator().remove();
    }

    @Override
    public void add(final Instance instance) {
        Sampler sampler = map.get(instance.classValue());
        if(sampler == null) {
            throw new UnsupportedOperationException();
        }
        sampler.getIterator().add(instance);
    }


}
