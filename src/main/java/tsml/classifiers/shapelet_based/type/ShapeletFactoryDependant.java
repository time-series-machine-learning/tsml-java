package tsml.classifiers.shapelet_based.type;

public class ShapeletFactoryDependant implements ShapeletFactoryMV {

    @Override
    public ShapeletMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double[][] instance) {
        int minLength = Integer.MAX_VALUE;
        for(int i=0;i<instance.length;i++){
            minLength = Math.min(minLength,instance[i].length);
        }
        ShapeletMV[] candidates = new ShapeletMV[minLength-shapeletSize];
        for (int seriesIndex=0;seriesIndex<minLength-shapeletSize;seriesIndex++){

            candidates[seriesIndex] = new ShapeletDependentMV(seriesIndex, shapeletSize, instanceIndex, instance);

        }
        return candidates;
    }


}
