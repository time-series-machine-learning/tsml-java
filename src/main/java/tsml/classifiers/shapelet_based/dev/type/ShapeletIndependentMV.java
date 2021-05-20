package tsml.classifiers.shapelet_based.dev.type;

import tsml.data_containers.TimeSeriesInstance;
import utilities.ClusteringUtilities;

public class ShapeletIndependentMV extends ShapeletSingle {

    private int seriesIndex;
    private double[] data;

    public ShapeletIndependentMV(int start, int length, int instanceIndex, double classIndex,  int seriesIndex, TimeSeriesInstance instance){
        super(start, length, instanceIndex, classIndex);
        this.seriesIndex = seriesIndex;
        this.setData(instance);
    }

    public int getSeriesIndex(){
        return seriesIndex;
    }



    public void setData(TimeSeriesInstance instance) {
            this.data = new double[length];
            for (int i=0;i<length;i++){
                    this.data[i] = instance.get(this.seriesIndex).get(start+i);
            }
        ClusteringUtilities.zNormalise(this.data);

    }

    public double[] getData(){
        return this.data;
    }

    @Override
    public double getDistanceToInstance(int start, TimeSeriesInstance instance) {
        double sum = 0;
        double temp = 0;
        double a,b,ab;
        for (int i = 0; i < length; i++)
        {
            temp = data[i] - instance.get(seriesIndex).get(start+i);

            sum = sum + (temp * temp);
        }
        return sum;
    }





    @Override
    public String toString(){
        return "Instance: " + instanceIndex + " Series: " + seriesIndex + " Start: " + start + " Length: " + length +
                 " Quality: " + quality + " Class "  + classIndex +  "\n";//->" + Arrays.toString(data) ;
    }




}
