package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;

public class DoubleToIntDistributionAdapter implements Distribution<Integer> {

    public Converter getConverter() {
        return converter;
    }

    public void setConverter(
        final Converter converter) {
        this.converter = converter;
    }

    public interface Converter {
        int convert(double value);
    }

    private Distribution<Double> distribution;
    private Converter converter;
    private static final Converter DEFAULT_CONVERTER = value -> {
        long rounded = Math.round(value);
        return (int) rounded;
    };

    public static Converter getDefaultConverter() {
        return DEFAULT_CONVERTER;
    }

    public DoubleToIntDistributionAdapter(
        final Distribution<Double> distribution,
        final Converter converter) {
        setConverter(converter);
        setDistribution(distribution);
    }

    public DoubleToIntDistributionAdapter(final Distribution<Double> distribution) {
        this(distribution, getDefaultConverter());
    }

    @Override
    public Integer sample() {
        final Double doubleSample = getDistribution().sample();
        final int converted = getConverter().convert(doubleSample);
        return converted;
    }

    @Override
    public void setRandom(final Random random) {
        distribution.setRandom(random);
    }

    @Override
    public Random getRandom() {
        return distribution.getRandom();
    }

    public Distribution<Double> getDistribution() {
        return distribution;
    }

    public void setDistribution(
        final Distribution<Double> distribution) {
        this.distribution = distribution;
    }
}
