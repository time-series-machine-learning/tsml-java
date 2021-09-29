package tsml.classifiers.shapelet_based.dev.type;

import tsml.data_containers.TimeSeriesInstance;

public class ShapeletDimensionDependentMV extends ShapeletDependentMV {

     int[] dimensionIndex;

    public ShapeletDimensionDependentMV(int start, int length, int instanceIndex, double classIndex, int[] dimensionIndex, TimeSeriesInstance instance) {
        super(start, length, instanceIndex, classIndex, instance);
        this.dimensionIndex = dimensionIndex;
        this.setDataDD(instance);
    }
    public void setDataDD( TimeSeriesInstance instance) {
        this.data = new double[dimensionIndex.length][this.length];
        for (int i=0;i<dimensionIndex.length;i++){
            for (int j=0;j<this.length;j++){
                this.data[i][j] = instance.get(dimensionIndex[i]).get(start+j);
            }
        }
    }

    public int getDimension(int d){
        return dimensionIndex[d];
    }
}
