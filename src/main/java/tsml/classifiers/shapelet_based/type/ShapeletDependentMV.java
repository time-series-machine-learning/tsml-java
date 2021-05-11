package tsml.classifiers.shapelet_based.type;

import tsml.data_containers.TimeSeriesInstance;

public  class ShapeletDependentMV extends ShapeletSingle{

     double[][] data;

    public ShapeletDependentMV(int start, int length, int instanceIndex, double classIndex, TimeSeriesInstance instance){
        super(start,length,instanceIndex,classIndex);
        this.setData(instance);
    }

    public void setData( TimeSeriesInstance instance) {
            this.data = new double[instance.getNumDimensions()][this.length];
            for (int i=0;i<instance.getNumDimensions();i++){
                for (int j=0;j<this.length;j++){
                    this.data[i][j] = instance.get(i).get(start+j);
                }
            }
    }

    public double[][] getData(){
        return this.data;
    }

    @Override
    public double getDistanceToInstance(int start, TimeSeriesInstance instance) {
        double sum = 0;
        double temp = 0;
        for(int channel=0;channel< instance.getNumDimensions(); channel++){

            for (int index = 0; index < length; index++)
            {
                temp = data[channel][index] - instance.get(channel).get(start+index);
                sum = sum + (temp * temp);
            }
        }
        return sum/data.length;
    }




    @Override
    public String toString(){
        return "Start: " + start + " Length: " + length + " Instance Index: " + instanceIndex
                 + " Quality: " + quality + "\n";//+  " " + Arrays.deepToString(data) + "\n";
    }


}
