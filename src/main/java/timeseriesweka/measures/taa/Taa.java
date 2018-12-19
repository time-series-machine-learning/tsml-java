package timeseriesweka.measures.taa;

import timeseriesweka.measures.DistanceMeasure;
import weka.core.TechnicalInformation;

public class Taa extends DistanceMeasure {

    private static final TAA_banded TAA = new TAA_banded();

    private int k;

    public Taa(int k, double gPenalty, double tPenalty) {
        this.k = k;
        this.gPenalty = gPenalty;
        this.tPenalty = tPenalty;
    }

    private double gPenalty;

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public double getGPenalty() {
        return gPenalty;
    }

    public void setGPenalty(double gPenalty) {
        this.gPenalty = gPenalty;
    }

    public double getTPenalty() {
        return tPenalty;
    }

    public void setTPenalty(double tPenalty) {
        this.tPenalty = tPenalty;
    }

    private double tPenalty;


    private int[] naturalNumbers(int size) {
        int[] numbers = new int[size];
        for(int i = 0; i < numbers.length; i++) {
            numbers[i] = i;
        }
        return numbers;
    }

    @Override
    protected double measureDistance(double[] timeSeriesA, double[] timeSeriesB, double cutOff) {
        return TAA.score(timeSeriesA,
                naturalNumbers(timeSeriesA.length),
                timeSeriesB,
                naturalNumbers(timeSeriesB.length),
                1, 1, 1);
    }

    @Override
    public String getRevision() {
        return null;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        return null;
    }

    public static void main(String[] args) {
        double[] a = new double[] {1,1,2,2,3,3,2,2,1,1};
        double[] b = new double[] {1,2,3,2,1,1,1,1,1,2};
        int[] aIntervals = new int[] {1,2,3,4,5,6,7,8,9,10};
        int[] bIntervals = new int[] {1,2,3,4,5,6,7,8,9,10};
        System.out.println(new TAA_banded().score(a,aIntervals,b,bIntervals,2,2,2));
        Taa taa = new Taa(2,2,2);
        System.out.println(taa.distance(a,b));
    }

    @Override
    public String getParameters() {
        return "k=" + k + ",tPenalty=" + tPenalty + ",gPenalty=" + gPenalty + ",";
    }

    @Override
    public String toString() {
        return "TAA";
    }
}
