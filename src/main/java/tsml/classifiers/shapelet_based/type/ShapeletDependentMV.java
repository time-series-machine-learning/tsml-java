package tsml.classifiers.shapelet_based.type;

import java.util.Arrays;

public  class ShapeletDependentMV extends ShapeletMV{

    private double[][] data;

    public ShapeletDependentMV(int start, int length, int instanceIndex, double[][] instance){
        super(start,length,instanceIndex);
        this.setData(instance);
    }

    public void setData(double[][] instance) {
            this.data = new double[instance.length][this.length];
            for (int i=0;i<instance.length;i++){
                for (int j=0;j<this.length;j++){
                    this.data[i][j] = instance[i][start+j];
                }
            }
    }

    @Override
    public double getDistanceToInstance(int start, double[][] instance) {
        double sum = 0;
        double temp = 0;
        for(int channel=0;channel< instance.length; channel++){

            for (int index = 0; index < length; index++)
            {
                temp = data[channel][index] - instance[channel][start+index];
                sum = sum + (temp * temp);
            }
        }
        return sum;
    }

    @Override
    public double[][] toDoubleArray() {
        return data;
    }

    @Override
    public String toString(){
        return "Start: " + start + " Length: " + length + " Instance Index: " + instanceIndex
                 + " Quality: " + quality +  " " + Arrays.deepToString(data) + "\n";
    }


}
