package weka.classifiers.rules.ruleshandler;

public class CItemCorpo {

    int[] item_corpo;


    public CItemCorpo(int k) {

        item_corpo = new int[k];

        for ( int i = 0 ; i < k ; i++ ) {
            item_corpo[i] = 0;
        }
    }

    public void ordina ( int left, int right) {

        int i = left;
        int j = right;

        int temp,pivot;

        pivot = item_corpo[(int)((left+right)/2)];

        do {
            while ( (item_corpo[i] < pivot) && ( i<right) ) {
                i++;
            }
            while ( (item_corpo[j] > pivot) && ( j>left ) ) {
                j--;
            }
            if ( i <= j ) {
                temp = item_corpo[i];
                item_corpo[i] = item_corpo[j];
                item_corpo[j] = temp;
                i++;
                j--;
            }

        } while ( i <= j);

        if ( left < j )
            ordina(left,j);
        if ( i < right )
            ordina(i,right);

        return;

    }

}
