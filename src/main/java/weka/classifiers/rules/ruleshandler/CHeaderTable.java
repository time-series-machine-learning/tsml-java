package weka.classifiers.rules.ruleshandler;

public class CHeaderTable {

    int frequentCount;
    CFrequentItem[] frequentArray;

    public CHeaderTable(int k) {
        frequentArray = new CFrequentItem[k];
        frequentCount = 0;
        for ( int t = 0 ; t<k ; t++) {
            frequentArray[t] = new CFrequentItem(CMain.MAX_CLASSES);
        }
    }

    public void quicksort(int left,int right) {

        CFrequentItem temp;
        CFrequentItem pivot;
        int i,j;

        i = left;
        j = right;

        pivot = frequentArray[(int)((left+right)/2)];

        do {
            while ( (itemCompare(frequentArray[i],pivot) < 0) && i<right ) {
                i++;
            }

            while ( (itemCompare(frequentArray[j],pivot) > 0) && j>left ) {
                j--;
            }
            if ( i <= j) {
                temp = frequentArray[i];
                frequentArray[i] = frequentArray[j];
                frequentArray[j] = temp;
                i++;
                j--;
            }

        } while ( i <= j );

        if ( left < j)
            quicksort(left , j);
        if ( right > i)
            quicksort(i, right);

        return;
    }


    public int itemCompare ( CFrequentItem cf1, CFrequentItem cf2) {


        if ( cf1.supp < cf2.supp )
            return -1;

        if ( cf1.supp > cf2.supp )
            return 1;

        if ( cf1.supp == cf2.supp) {

            if ( cf1.itemId < cf2.itemId )
                return -1;
            if ( cf1.itemId > cf2.itemId )
                return 1;

        }

        return 0;
    }


    public void setFrequent ( int j) {
        frequentCount = j;
    }
}
