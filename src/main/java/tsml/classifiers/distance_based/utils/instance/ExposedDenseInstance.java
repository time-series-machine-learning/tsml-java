package tsml.classifiers.distance_based.utils.instance;

import weka.core.DenseInstance;
import weka.core.Instance;

public class ExposedDenseInstance extends DenseInstance {

    public ExposedDenseInstance(final Instance instance) {
        super(instance);
    }

    @Override
    public double[] toDoubleArray() {
        // directly exposes double[] to outsiders
        return m_AttValues;
    }

    public static double[] extractAttributeValuesAndClassLabel(Instance instance) {
        final ExposedDenseInstance exposedDenseInstance = new ExposedDenseInstance(instance);
        return exposedDenseInstance.toDoubleArray();
    }
}
