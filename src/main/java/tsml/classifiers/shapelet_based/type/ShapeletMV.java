package tsml.classifiers.shapelet_based.type;

import tsml.data_containers.TimeSeriesInstance;

public abstract class ShapeletMV implements Comparable<ShapeletMV>{

    protected int start;
    protected int length;
    protected int instanceIndex;
    protected double quality;

    public double getClassIndex() {
        return classIndex;
    }

    public void setClassIndex(double classIndex) {
        this.classIndex = classIndex;
    }

    protected double classIndex;

    public ShapeletMV(int start, int length, int instanceIndex, double classIndex){
        this.start = start;
        this.length = length;
        this.instanceIndex = instanceIndex;
        this.classIndex = classIndex;
    }

    protected abstract void setData(double[][] instance);

    public abstract double getDistanceToInstance(int start, TimeSeriesInstance instance);


    public void setQuality(double quality){
        this.quality = quality;
    }
    public double getQuality(){
        return this.quality;
    }

    public int getLength(){
        return length;
    }

    @Override
    public int compareTo(ShapeletMV shapeletMV) {
        return (this.getQuality()>shapeletMV.getQuality()?-1:(this.getQuality()<shapeletMV.getQuality()?1:0));
    }
}
