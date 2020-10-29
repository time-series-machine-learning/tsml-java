package tsml.classifiers.distance_based.utils.collections.params.distribution;

import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class CompositeDistribution<A> extends BaseDistribution<A> {

    public static <A> CompositeDistribution<A> newCompositeFromRange(List<A> list, BiFunction<A, A, Distribution<A>> func) {
        final CompositeDistribution<A> distribution = new CompositeDistribution<>();
        for(int i = 0; i < list.size() - 1; i++) {
            final A start = list.get(i);
            final A end = list.get(i + 1);
            distribution.getDistributions().add(func.apply(start, end));
        }
        return distribution;
    }
    
    public static CompositeDistribution<Double> newUniformDoubleCompositeFromRange(List<Double> list) {
        return newCompositeFromRange(list, UniformDoubleDistribution::new);
    }
    
    public static CompositeDistribution<Integer> newUniformIntCompositeFromRange(List<Integer> list) {
        return newCompositeFromRange(list, UniformIntDistribution::new);
    }
    
    public CompositeDistribution() {}
    
    public CompositeDistribution(Distribution<A>... distributions) {
        this(newArrayList(distributions));
    }
    
    public CompositeDistribution(List<Distribution<A>> distributions) {
        setDistributions(distributions);
    }
    
    private List<Distribution<A>> distributions = new ArrayList<>();
    
    @Override public A sample() {
        final Random random = getRandom();
        final int index = random.nextInt(distributions.size());
        final Distribution<A> distribution = distributions.get(index);
        return distribution.sample(random);
    }

    public List<Distribution<A>> getDistributions() {
        return distributions;
    }

    public void setDistributions(final List<Distribution<A>> distributions) {
        this.distributions = distributions;
    }
}
