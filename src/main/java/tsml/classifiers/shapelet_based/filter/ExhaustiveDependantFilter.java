package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.classifiers.ShapeletInterface;
import tsml.classifiers.shapelet_based.classifiers.ShapeletDataMV;
import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;

import java.util.ArrayList;

public class ExhaustiveDependantFilter implements ShapeletFilterMV {


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
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances, ShapeletQuality quality) {
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        int index =0;
        for (TimeSeriesInstance ts: instances){

            int minLength = Integer.MAX_VALUE;
            for(TimeSeries series: ts){
                minLength = Math.min(minLength,series.getSeriesLength());
            }

            for (int i=params.min;i<=params.max;i++){
                for (int j=0;j<minLength;j++){
                    ShapeletMV candidate = new ShapeletMV(j,j+i, index, -1);
                    candidate.setQuality(0.0);
                }
            }
            index++;
        }
        return shapelets;
    }

    /*private double getQuality(ShapeletDataMV candidate, TimeSeriesInstances instances,
                              ShapeletQualityMV quality){
        OrderLineObj[] orderline = new OrderLineObj[instances.numInstances()];

        for (int i=0;i< orderline.length;i++){
            orderline[i] = new OrderLineObj(
                    candidate.distanceByIndex(instances.get(i).get(0),0),
                    instances.get(i).getLabelIndex()
            );


        }
        return quality.getQuality(orderline);
    }
*/

}
