package evaluation.tuning;

import weka.core.Instances;

public abstract class ParameterSpaceFromInstancesBuilder extends ParameterSpaceBuilder {
    public Instances getInstances() {
        return instances;
    }

    public void setInstances(final Instances instances) {
        this.instances = instances;
    }

    private Instances instances;
}
