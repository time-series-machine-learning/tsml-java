package classifiers.template.configuration;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class Restricted {
    private Instances instances;
    private final List<Instance> restrictedInstances = new ArrayList<>();

    public List<Instance> getRestrictedInstances() {
        return restrictedInstances;
    }

    public Instances getInstances() {
        return instances;
    }

    public void setInstances(final Instances instances) {
        this.instances = instances;
    }
}
