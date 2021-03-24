package tsml.classifiers.shapelet_based.type;

public interface ShapeletFactoryMV {
    ShapeletMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, double[][] instance);
    ShapeletMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, double[][] instance);
}
