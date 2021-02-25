package tsml.classifiers.shapelet_based.type;

public abstract class ShapeletMV implements Comparable<ShapeletMV>{

    protected int start;
    protected int length;
    protected int instanceIndex;
    protected double quality;

    public ShapeletMV(int start, int length, int instanceIndex){
        this.start = start;
        this.length = length;
        this.instanceIndex = instanceIndex;
    }

    protected abstract void setData(double[][] instance);

    public abstract double getDistanceToInstance(int start, double[][] instance);
    public abstract double[][] toDoubleArray();

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
