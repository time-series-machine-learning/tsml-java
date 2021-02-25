package tsml.classifiers.shapelet_based.type;

public interface ShapeletFactoryMV {
    ShapeletMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double[][] instance);
}
