package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.classifiers.ShapeletDataMV;
import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityMV;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityMeasureMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;

import java.util.ArrayList;
import java.util.Arrays;

public class ExhaustiveIndependantFilter implements ShapeletFilterMV {


  /*  protected ArrayList<ShapeletMV> getShapeletsOnSingleSeries(TimeSeriesInstances instances){
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        for (TimeSeriesInstance instance : instances){
            TimeSeries series = instance.get(0);
            for (int size = min;size<= max; size++){
                for (int i=0;i<series.getSeriesLength()-size;i++){
                    ShapeletDataMV shapelet = createShapelet(series, i,size);
                 //   double quality = getQuality(shapelet,instances)
                }
            }
        }
        return shapelets;
    }
    private ShapeletDataMV createShapelet(TimeSeries series, int start, int size){
        ShapeletDataMV shapelet = new ShapeletDataMV(size,1);
        for (int i=0;i<size;i++){
            shapelet.setData(series.get(i+start),i,0);
        }
        return shapelet;
    }
*/
    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances,
                                               ShapeletDistanceMV distance, ShapeletQualityMeasureMV quality) {
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        double[][][] instancesArray = instances.getHSliceArray(instances.getClassIndexes());
        for (int index=0;index<instancesArray.length;index++){
           int k=0;
            for(int k=0;k<instancesArray[index].length;k++){
                for (int j=0;j<instancesArray[index][k].length;j++){
                    for (int i=params.min;i<=params.max;i++) {

                        ShapeletMV candidate = new ShapeletMV(j, j + i, index, k, instancesArray[index]);
                        candidate.setQuality(getQuality(candidate,instancesArray,instances.getClassIndexes(),
                                int[] classCounts,
                                instancesArray[index][k].length, distance,quality
                                ));
                    }
                }
                k++;
            }

        }
        return shapelets;
    }

    private double getQuality(ShapeletMV candidate, double[][][] instancesArray, int[] classIndexes, int[] classCounts, int length, ShapeletDistanceMV distance, ShapeletQualityMeasureMV quality) {
        OrderLineObj[] orderline = new OrderLineObj[instances.length];

        for (int i=0;i< orderline.length;i++){
            double dist = distance.distance(candidate,instancesArray[i],length);
            orderline[i] = new OrderLineObj(
                   dist,classIndexes[i]);

        }
        return quality.calculate(Arrays.asList(orderline), new ClassCounts )
    }



}
