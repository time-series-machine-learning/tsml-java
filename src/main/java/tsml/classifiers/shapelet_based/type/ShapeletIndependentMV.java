package tsml.classifiers.shapelet_based.type;

import tsml.data_containers.TimeSeriesInstance;

public class ShapeletIndependentMV extends ShapeletMV {

    private int seriesIndex;
     double[] data;

    public ShapeletIndependentMV(int start, int length, int instanceIndex, double classIndex,  int seriesIndex, double[][] instance){
        super(start, length, instanceIndex, classIndex);
        this.seriesIndex = seriesIndex;
        this.setData(instance);
    }

    public void setData(double[][] instance) {
            this.data = new double[length];
            for (int i=0;i<length;i++){
                    this.data[i] = instance[this.seriesIndex][start+i];
            }

    }

    @Override
    public double getDistanceToInstance(int start, TimeSeriesInstance instance) {
        double sum = 0;
        double temp = 0;
        for (int i = 0; i < length; i++)
        {
            temp = data[i] - instance.get(seriesIndex).get(start+i);

            sum = sum + (temp * temp);
        }
        return sum;
    }



    @Override
    public String toString(){
        return "Start: " + start + " Length: " + length + " Instance Index: " + instanceIndex
                + " Channel Index: " + seriesIndex + " Quality: " + quality + " Class "  + classIndex +  "\n";//->" + Arrays.toString(data) ;
    }




}
