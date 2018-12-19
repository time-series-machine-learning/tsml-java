package weka.classifiers.rules.ruleshandler;

public class CMacroItem {

    int numItem;
    int[] vettItemId;
    static int MACROITEM_MAX_ITEM = 3000 ;


    public CMacroItem() {
        numItem = 0;
        vettItemId = new int[MACROITEM_MAX_ITEM];

    }
}
