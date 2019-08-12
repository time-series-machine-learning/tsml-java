package evaluation.tuning;

import weka.core.Instances;

public abstract class ParameterSpaceFromInstancesBuilder extends ParameterSpaceBuilder {
    public Instances getInstances() {
        return instances;
    }

    public ParameterSpaceFromInstancesBuilder setInstances(final Instances instances) {
        this.instances = instances;
        return this;
    }

    private Instances instances;

    public ParameterSpace build(Instances instances) {
        setInstances(instances);
        return build();
    }
}
