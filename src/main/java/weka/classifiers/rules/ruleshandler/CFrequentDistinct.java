package weka.classifiers.rules.ruleshandler;

public class CFrequentDistinct {

    int freqdistinct;

    public CFrequentDistinct(int val) {

        freqdistinct = val;
    }

    public void modifica(int mod) {

        freqdistinct = mod;
    }

    public void add(int agg) {
        freqdistinct = freqdistinct + agg;
    }

}
