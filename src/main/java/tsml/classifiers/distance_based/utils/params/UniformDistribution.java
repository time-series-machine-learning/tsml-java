package tsml.classifiers.distance_based.utils.params;

class UniformDistribution extends Distribution {

    @Override
    public double sample() {
        return getRandom().nextDouble();
    }
}
