package weka.classifiers.rules.ruleshandler;

public class CItem {

    int supp;
    int[] suppClass;

    public CItem(int k) {

        supp = 0;
        suppClass = new int[k];
        for ( int i = 0 ; i<k ; i++ ) {
            suppClass[i] = 0;
        }
    }

}
