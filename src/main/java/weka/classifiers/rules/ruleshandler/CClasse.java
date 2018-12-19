package weka.classifiers.rules.ruleshandler;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.FieldPosition;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Locale;


public class CClasse {



    double[] macroRegoleClassi;
    double[] regoleClassi;


    static int MAX_CLASSES = 30; // Da modificare anche nel main
    static int MAX_ITEM = 100000;
    static int RULE_MAX_LENGHT = 1300;
    static int CODA_MAX_LEN = 1300;

    int classeDesiderata;
    int maxNumItem;
    double tidSet;
    double regoleTotali;
    double genTotali;
    double maxNodi;
    double maxRec;
    double totRec;
    double maxDepth;
    double macroRegoleTotali;


    static int CONDITION_MAX_LEN = 3000;
    LinkedList<CFpNode> nodi;
    CFrequentItem node_pattern_base;
    double noditree;
    double create_node;

    StringBuffer s = new StringBuffer();

    public CClasse () {

        macroRegoleClassi = new double[MAX_CLASSES];
        regoleClassi = new double[MAX_CLASSES];

        nodi = new LinkedList<CFpNode>();

    }

    public void estraiPerClasse(int len,String[] arg,int classeAttuale) {

        CItem[] tmptab ;
        int supp_thres;
        CFrequentDistinct frequentcounter = new CFrequentDistinct(0);
        CFrequentDistinct distinctcounter = new CFrequentDistinct(0);
        CHeaderTable htab ;
        CFptree fp ;
        CMacroItem[] condition = new CMacroItem[CONDITION_MAX_LEN];
        CMacroItem[] coda = new CMacroItem[CODA_MAX_LEN];
        String nomeFileUscita;
        PrintWriter piw = null;

        classeDesiderata = classeAttuale ;
        regoleTotali = 0;
        genTotali = 0;
        macroRegoleTotali = 0;
        maxNodi = 0;
        maxRec = 0;
        totRec = 0;
        maxDepth = 0;

        for ( int h = 0 ; h < CODA_MAX_LEN ; h++ ) {
            coda[h] = new CMacroItem();
        }

        for ( int h = 0 ; h < CONDITION_MAX_LEN ; h++ ) {
            condition[h] = new CMacroItem();
        }


        supp_thres = (((int)CMain.supp_threshold)*CMain.suppClasses[classeDesiderata])/100;

        if ( supp_thres < 1) {
            supp_thres = 1;
        }

        if ( (tmptab = supportCounting(arg[0],supp_thres,frequentcounter,distinctcounter)) == null ) {
            System.out.println("Not create temp table\n");
            System.exit(3);
        }

        if ( (htab = firstHeaderTableCreate(tmptab,supp_thres,frequentcounter)) == null) {

            System.out.println("Not create htable\n");
            System.exit(4);
        }


        if ( (fp = firstFpTreeCreate(arg[0],htab,distinctcounter.freqdistinct)) == null) {

            System.out.println("Not create fp tree");
            System.exit(5);
        }

        nomeFileUscita = new String(arg[4]+"c"+classeDesiderata+arg[3]);

        try {

            piw = new PrintWriter( new BufferedWriter ( new FileWriter ( nomeFileUscita)));

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }


        fpMine ( htab,fp,condition,1,supp_thres,piw,coda,0,supp_thres,0);

        macroRegoleClassi[classeAttuale]  = macroRegoleTotali;
        regoleClassi[classeAttuale] = regoleTotali;

        return;
    }

    public CItem[] supportCounting(String fileName,int threshold,CFrequentDistinct frequentcounter,CFrequentDistinct distinctcounter) {

        CItem[] tmptab = new CItem[MAX_ITEM];

        for ( int z=0 ; z<MAX_ITEM ; z++) {
            tmptab[z] = new CItem(MAX_CLASSES);
        }

        byte b;
        int n=0;
        int[] t ;

        try {

            FileInputStream fis = new FileInputStream ( fileName );

            DataInputStream di = new DataInputStream(fis);

            maxNumItem = 0;

            while ( true ) {
                for ( int h = 0 ; h<3 ; h++ ) {

                    ByteBuffer bf = ByteBuffer.allocate(4);
                    for ( int k = 0 ; k<4 ; k++ ) {
                        b = di.readByte();

                        bf.order(ByteOrder.LITTLE_ENDIAN);

                        bf.put(b);
                    }
                    n = bf.getInt(0);
                }


                t = new int[n];

                for ( int f = 0 ; f<n ; f++) {
                    ByteBuffer buf = ByteBuffer.allocate(4);
                    for ( int g = 0 ; g<4 ; g++ ) {
                        b = di.readByte();

                        buf.order(ByteOrder.LITTLE_ENDIAN);

                        buf.put(b);

                    }
                    t[f] = buf.getInt(0);
                }
                for ( int k = 0 ; k<n-1 ; k++ ) {
                    if ( tmptab[t[k]-1].supp == 0 ) {
                        distinctcounter.add(1);
                        if ( t[k] > maxNumItem )
                            maxNumItem = t[k];
                    }
                    tmptab[t[k]-1].supp++;
                    tmptab[t[k]-1].suppClass[t[n-1]-CMain.idBaseClasse]++;

                    if ( (t[n-1]-CMain.idBaseClasse) == classeDesiderata ) {
                        if ( tmptab[t[k]-1].suppClass[classeDesiderata] == threshold ) {
                            frequentcounter.add(1);
                        }
                    }

                }

            }

        } catch (EOFException eofx) {

            maxNumItem++;

            return tmptab ;

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        return null;
    }

    public CHeaderTable firstHeaderTableCreate(CItem[] tmptab,int supp_thres,CFrequentDistinct frequentcounter) {

        CHeaderTable htab = new CHeaderTable(frequentcounter.freqdistinct);

        htab.frequentCount = frequentcounter.freqdistinct;

        int j = 0;

        for ( int i = 0 ; i<maxNumItem ; i++ ) {

            if ( tmptab[i].suppClass[classeDesiderata] >= supp_thres) {
                CFrequentItem cfi = new CFrequentItem(MAX_CLASSES);
                cfi.itemId = i+1;
                cfi.supp = tmptab[i].supp;

                for ( int h = 0 ; h<MAX_CLASSES ; h++) {
                    cfi.suppClass[h] = tmptab[i].suppClass[h];
                }
                htab.frequentArray[j] = cfi;
                j++;
            }
        }
        if (frequentcounter.freqdistinct > 0) /***riga aggiunta ***/
            htab.quicksort(0,frequentcounter.freqdistinct-1);

        return htab;
    }

    public CFptree firstFpTreeCreate(String file,CHeaderTable htab,int distinctcounter) {

        CFptree fpt = new CFptree(MAX_CLASSES);
        CFpNode parent;
        CFpNode current;
        int itemClasse;
        int[] present_item;
        byte b;
        int n=0;
        int itemid;


        try {


            FileInputStream fis = new FileInputStream ( file );

            DataInputStream di = new DataInputStream(fis);

            present_item = new int [distinctcounter];

            while ( true ) {
                for ( int h = 0 ; h<3 ; h++ ) {

                    ByteBuffer bf = ByteBuffer.allocate(4);
                    for ( int k = 0 ; k<4 ; k++ ) {
                        b = di.readByte();

                        bf.order(ByteOrder.LITTLE_ENDIAN);

                        bf.put(b);
                    }
                    n = bf.getInt(0);

                }

                for ( int a = 0 ; a<distinctcounter-1 ; a++ ) {
                    present_item[a] = 0;
                }

                for ( int u = 0 ; u < n-1 ; u++ ) {
                    ByteBuffer buf = ByteBuffer.allocate(4);
                    for ( int g = 0 ; g<4 ; g++ ) {

                        b = di.readByte();

                        buf.order(ByteOrder.LITTLE_ENDIAN);

                        buf.put(b);

                    }

                    itemid = buf.getInt(0);

                    present_item[itemid-1] = 1;

                }

                ByteBuffer buf2 = ByteBuffer.allocate(4);
                for ( int g = 0 ; g<4 ; g++ ) {

                    b = di.readByte();

                    buf2.order(ByteOrder.LITTLE_ENDIAN);

                    buf2.put(b);

                }

                itemClasse = buf2.getInt(0);

                itemClasse = itemClasse-CMain.idBaseClasse;

                if ( itemClasse < 0 ) {
                    System.out.println("Classe errata "+itemClasse);
                    System.exit(1);
                }

                parent = fpt.root;

                for ( int i = htab.frequentCount-1 ; i >=0 ; i-- ) {
                    if ( present_item[htab.frequentArray[i].itemId-1] == 1 ) {
                        present_item[htab.frequentArray[i].itemId-1] = 0;

                        if ( (current = firstInsertNode(parent,htab.frequentArray[i],itemClasse,i)) == null ) {
                            return null;
                        }
                        parent = current;
                    }

                }
            }


        } catch (EOFException eofx) {

            return fpt;

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        return null;
    }

    public CFpNode firstInsertNode(CFpNode parent,CFrequentItem htab_entry,int item_classe,int index) {

        CChildPtr curr_elem;
        CChildPtr prev_elem;
        CChildPtr tmp_elem;

        CFpNode tmp_child;
        int found = 0;

        curr_elem = parent.children;
        prev_elem = parent.children;


        while ((curr_elem != null) && (found == 0 ) ) {

            if ( curr_elem.child.itemId == htab_entry.itemId) {

                found = 1;

            } else {
                prev_elem = curr_elem;
                curr_elem = curr_elem.next;

            }

        }

        if ( found == 1) {

            curr_elem.child.local_supp++;
            curr_elem.child.local_suppClass[item_classe]++;
            parent = curr_elem.child;
        }

        else {
            tmp_elem = new CChildPtr(MAX_CLASSES);
            tmp_child = tmp_elem.child;
            tmp_child.itemId = htab_entry.itemId;
            tmp_child.parent = parent;
            tmp_child.local_supp = 1;

            for ( int x = 0 ; x < MAX_CLASSES ; x++) {
                tmp_child.local_suppClass[x] = 0;
            }

            tmp_child.local_suppClass[item_classe] = 1;

            tmp_child.next = htab_entry.head;

            if ( prev_elem == null ) {

                parent.children = tmp_elem;
            } else {
                prev_elem.next = tmp_elem;
            }
            htab_entry.head = (tmp_child);
            htab_entry.incCounter(1);

            parent = tmp_child;

        }

        return parent;

    }

    public void fpMine (CHeaderTable htab,CFptree fptree,CMacroItem[] condition,int minedepth,int supp_thres,PrintWriter nomeFileUscita,CMacroItem[] coda,int dimcoda,int supp_cond,double noditotali) {

        CFrequentItem[] freq_array ;
        CFpNode node;
        CFpNode nodelink;
        CFptree new_fptree;
        CHeaderTable new_htab;

        CItemEntry tmptab = new CItemEntry(MAX_CLASSES);
        CItemEntry new_entry = new CItemEntry(MAX_CLASSES);
        CItemEntry prev_entry = new CItemEntry(MAX_CLASSES);
        CItemEntry curr_entry = new CItemEntry(MAX_CLASSES);
        int found;
        int oldsup;
        totRec++;
        int i;
        int frequent_count = 0;
        int t = 0;

        if ( minedepth > 1 ) {
            dimcoda = accresciCoda(coda,dimcoda,htab,supp_cond);
        }

        if ( fptree.branches == 1 ) {
            combineItem(htab,condition,minedepth,htab.frequentCount,nomeFileUscita,supp_thres,coda,dimcoda);
        } else {
            freq_array = htab.frequentArray;

            for (  i = 0; i<htab.frequentCount ; i++) {
                if ( freq_array[i].accorpato == 0) {

                    if ( freq_array[i].nodeLinkCounter == 0) {
                        System.err.println("Errore: freq_array["+i+"].nodeLinkCounter è zero!!");
                        return;
                    }
                    int sitem = 1;
                    condition[minedepth-1] = new CMacroItem();
                    condition[minedepth-1].vettItemId[0] = freq_array[i].itemId;

                    Iterator<Integer> it = freq_array[i].itemAccorpati.iterator();
                    while (it.hasNext()) {
                        int tmp = it.next();
                        condition[minedepth-1].vettItemId[sitem] = tmp;
                        sitem++;

                    }
                    condition[minedepth -1].numItem = sitem;

                    frequent_count = 0;
                    tmptab = null;
                    new_htab = null;
                    new_fptree = null;
                    nodelink = freq_array[i].head;

                    for ( int q = 0 ; q<freq_array[i].nodeLinkCounter ; q++) {
                        node = nodelink.parent;
                        while ( node.parent != null ) {
                            for ( t = i+1; t<(htab.frequentCount) && (freq_array[t].itemId!=node.itemId); t++) {}

                            if ( (t < htab.frequentCount) && (freq_array[t].accorpato == 0 )) {
                                curr_entry = tmptab;
                                prev_entry = tmptab;
                                found = 0;
                                while ( (curr_entry != null) && (found == 0) ) {
                                    if ( curr_entry.itemId == node.itemId) {
                                        found = 1;
                                        oldsup = curr_entry.suppClass[classeDesiderata];
                                        curr_entry.supp += nodelink.local_supp;
                                        for ( int u = 0 ; u<MAX_CLASSES ; u++) {
                                            curr_entry.suppClass[u] += nodelink.local_suppClass[u];
                                        }
                                        if ( (oldsup < supp_thres ) && (curr_entry.suppClass[classeDesiderata] >= supp_thres ) ) {
                                            frequent_count++;
                                        }
                                    } else {
                                        prev_entry = curr_entry;
                                        curr_entry = curr_entry.next;
                                    }
                                }
                                if ( found == 0 ) {
                                    new_entry = new CItemEntry(MAX_CLASSES);
                                    new_entry.itemId = node.itemId;
                                    new_entry.supp = nodelink.local_supp;

                                    for ( int g = 0 ; g<MAX_CLASSES ; g++ ) {
                                        new_entry.suppClass[g] = nodelink.local_suppClass[g];
                                    }
                                    if (new_entry.suppClass[classeDesiderata] >= supp_thres) {
                                        frequent_count++;
                                    }
                                    if ( prev_entry == null ) {
                                        tmptab = new_entry;
                                    } else {
                                        prev_entry.next = new_entry;
                                    }
                                }
                            }
                            node = node.parent;
                        }
                        nodelink = nodelink.next;

                    }

                    if ( frequent_count == 0) {
                        storeItemset(condition,minedepth,null,0,freq_array[i],0,0,null,nomeFileUscita,supp_thres,coda,dimcoda);
                    } else {
                        if ( (new_htab = headerTableCreate(tmptab,supp_thres,frequent_count)) == null ) {
                            System.exit(1);
                        }
                        for ( int w = 0; w<new_htab.frequentCount ; w++ ) {
                            int hti = 0;
                            while ( freq_array[hti].itemId != new_htab.frequentArray[w].itemId) {
                                hti++;
                            }
                            Iterator<Integer> iter = freq_array[hti].itemAccorpati.iterator();
                            while ( iter.hasNext()) {
                                new_htab.frequentArray[w].itemAccorpati.addFirst(iter.next());
                            }
                        }

                        int newdimcoda = dimcoda;
                        newdimcoda = accresciCodaperCondition(coda,newdimcoda,new_htab,freq_array[i].supp);
                        storeItemset(condition, minedepth, null, 0 , freq_array[i], 0 , 0, null, nomeFileUscita, supp_thres, coda, newdimcoda);


                        node_pattern_base = freq_array[i];

                        if ( (new_fptree = fptCreate(new_htab)) == null) {
                            System.exit(1);
                        }

                        freq_array[i] = node_pattern_base;

                        noditotali = noditotali + noditree;

                        if ( noditotali > maxNodi ) {
                            maxNodi = noditotali;
                            maxRec = minedepth;
                        }

                        fpMine(new_htab, new_fptree, condition, minedepth+1, supp_thres, nomeFileUscita, coda, dimcoda, freq_array[i].supp, noditotali);

                    }
                }
            }
        }
        return;
    }

    public int accresciCoda(CMacroItem[] coda,int dimcoda,CHeaderTable htab,int supp_cond) {

        int newfreqcounter;
        int ht;
        int sitem;

        CFrequentItem[] freq_array;

        freq_array = htab.frequentArray;

        for ( ht = 0 ; ht<htab.frequentCount ; ht++ ) {
            if ( freq_array[ht].supp == supp_cond ) {
                freq_array[ht].accorpato = 1;
                sitem = 1;
                coda[dimcoda].vettItemId[0] = freq_array[ht].itemId;

                Iterator<Integer> it = freq_array[ht].itemAccorpati.iterator();
                while ( it.hasNext() ) {
                    coda[dimcoda].vettItemId[sitem] = it.next();
                    sitem++;
                }
                coda[dimcoda].numItem = sitem;
                dimcoda++;
            }
        }
        newfreqcounter = 0;
        for ( ht=0 ; ht<htab.frequentCount ; ht++ ) {
            if ( freq_array[ht].accorpato == 0 ) {
                freq_array[newfreqcounter] = freq_array[ht];
                newfreqcounter++;
            }

        }
        htab.setFrequent(newfreqcounter);

        return dimcoda;
    }

    public void creaMacroItem (CHeaderTable htab,CFptree fptree) {

        CFrequentItem[] freq_array;
        int ih,ht;
        int newFreqcounter;
        int rif,c,p;
        int dim_possibili_item_eq;
        int uguali;
        CItemEq[] possibili_item_eq = new CItemEq[maxNumItem];

        for ( int j = 0 ; j<maxNumItem ; j++ ) {
            possibili_item_eq[j] = new CItemEq();
        }

        freq_array = htab.frequentArray;

        for ( ih = htab.frequentCount-1 ; ih >= 0 ; ih-- ) {
            freq_array[ih].accorpato = 0;
        }

        for ( rif = htab.frequentCount-1 ; rif>0 ; rif-- ) {
            dim_possibili_item_eq = 0;
            if ( freq_array[rif].accorpato == 0 ) {
                ih = rif-1;
                while ( (ih > 0) && (freq_array[ih].supp == freq_array[rif].supp) ) {
                    if ( freq_array[ih].accorpato == 0 ) {
                        uguali = 1;
                        c = 0;
                        while ( (c<MAX_CLASSES) && (uguali == 1) ) {
                            if ( freq_array[ih].suppClass[c] != freq_array[rif].suppClass[c] ) {
                                uguali = 0;
                            }
                            c++;
                        }
                        if ( uguali == 1 ) {
                            possibili_item_eq[dim_possibili_item_eq].pos = ih;
                            possibili_item_eq[dim_possibili_item_eq].flag = 0;
                            dim_possibili_item_eq++;
                        }
                    }
                    ih--;
                }
                if ( dim_possibili_item_eq > 0 ) {
                    equivalenti(htab,fptree,rif,possibili_item_eq,dim_possibili_item_eq);

                    for ( p = 0 ; p < dim_possibili_item_eq ; p++ ) {
                        if ( possibili_item_eq[p].flag == 0 ) {
                            freq_array[possibili_item_eq[p].pos].accorpato = 1;

                            int itemacc = freq_array[possibili_item_eq[p].pos].itemId;
                            freq_array[rif].itemAccorpati.addFirst(itemacc);
                            Iterator<Integer> it = freq_array[possibili_item_eq[p].pos].itemAccorpati.iterator();

                            while ( it.hasNext() ) {
                                itemacc = it.next();
                                freq_array[rif].itemAccorpati.addFirst(itemacc);
                            }


                        }
                    }
                }
            }
        }

        freq_array = htab.frequentArray;

        newFreqcounter = 0;

        for ( ht = 0 ; ht < htab.frequentCount ; ht++ ) {
            if ( freq_array[ht].accorpato == 0 ) {
                freq_array[newFreqcounter] = freq_array[ht];
                newFreqcounter++;
            } else {
                freq_array[ht].itemAccorpati.clear();
            }
        }

        htab.setFrequent(newFreqcounter);

        return;
    }

    public void equivalenti(CHeaderTable htab,CFptree fptree,int pos_item_di_rif ,CItemEq[] possibili_item_eq, int dim_possibili_item_eq ) {

        CItem[] itempresenti = new CItem[maxNumItem];
        CFpNode nodelink;
        int esistono_candidati;
        int i,c,livello;
        CChildPtr curr_elem;

        for ( int k = 0 ; k < maxNumItem ; k++ ) {
            itempresenti[k] = new CItem(MAX_CLASSES);
        }

        esistono_candidati = 1;

        nodelink = htab.frequentArray[pos_item_di_rif].head;

        livello = pos_item_di_rif-possibili_item_eq[dim_possibili_item_eq-1].pos;

        while ( (nodelink != null) && ( esistono_candidati == 1 ) ) {
            curr_elem = nodelink.children;

            while ( curr_elem != null ) {
                aggiornaPresenze ( curr_elem , itempresenti , livello );

                curr_elem = curr_elem.next;
            }
            esistono_candidati = 0;

            for ( i = 0 ; i < dim_possibili_item_eq ; i++ ) {
                if ( (possibili_item_eq[i].flag == 0) && ( itempresenti[(htab.frequentArray[possibili_item_eq[i].pos]).itemId].supp == nodelink.local_supp) ) {
                    c = 0;
                    while ( ( c < MAX_CLASSES ) && ( itempresenti[(htab.frequentArray[possibili_item_eq[i].pos]).itemId].suppClass[c] == nodelink.local_suppClass[c])) {
                        c++;
                    }

                    if ( c == MAX_CLASSES ) {
                        esistono_candidati = 1;
                    } else {
                        possibili_item_eq[i].flag = -1;
                    }
                } else {
                    possibili_item_eq[i].flag = -1;
                }
            }
            nodelink = nodelink.next;
        }

        return;
    }

    public void aggiornaPresenze ( CChildPtr curr_elem ,CItem[] itempresenti , int livello ) {

        CFpNode node;
        CChildPtr child;
        int c;
        node = curr_elem.child;
        itempresenti[node.itemId].supp = itempresenti[node.itemId].supp+node.local_supp;

        for ( c = 0 ; c < MAX_CLASSES ; c++ ) {
            itempresenti[node.itemId].suppClass[c] = itempresenti[node.itemId].suppClass[c] + node.local_suppClass[c];
        }

        child = node.children;

        while ( (child != null) && ( livello > 1 ) ) {
            aggiornaPresenze(child, itempresenti, livello-1);
            child = child.next;
        }

        return;
    }

    public void combineItem(CHeaderTable htab, CMacroItem[] condition,int minedepth,int comb_size,PrintWriter file,int supp_thres,CMacroItem[] coda,int dimcoda) {

        CMacroItem[] comb = new CMacroItem[comb_size] ;
        CFrequentItem[] freq_array;

        for ( int t = 0 ; t<comb_size ; t++ ) {
            comb[t] = new CMacroItem();
        }

        int sitem ;
        if ( htab == null )
            return;

        if ( htab.frequentCount == 0)
            return;

        freq_array = htab.frequentArray;

        int cl = 0;

        for ( int ht=htab.frequentCount-1 ; ht >= 0 ; ht-- ) {
            sitem = 1;
            condition[minedepth-1].vettItemId[0] = freq_array[ht].itemId;
            int pos = 0;
            while ( pos != freq_array[ht].itemAccorpati.size()) {
                condition[minedepth-1].vettItemId[sitem] = freq_array[ht].itemAccorpati.get(pos);
                sitem++;
                pos++;
            }
            condition[minedepth-1].numItem = sitem;
            storeItemset(condition, minedepth, comb, cl, freq_array[ht], 0, 0, null, file, supp_thres, coda, dimcoda);
            sitem = 1;
            pos = 0;
            comb[cl].vettItemId[0] = freq_array[ht].itemId;
            while ( pos != freq_array[ht].itemAccorpati.size()) {
                comb[cl].vettItemId[sitem] = freq_array[ht].itemAccorpati.get(pos);
                sitem++;
                pos++;
            }
            comb[cl].numItem = sitem;
            cl++;
        }

        return;
    }


    public void storeItemset(CMacroItem[] condition,int condition_lenght ,CMacroItem[] comb,int comb_size,CFrequentItem items ,int position ,int store_level,CMacroItem[] itemset,PrintWriter piw,int supp_thres,CMacroItem[] coda,int dimcoda) {

        double regole_rappresentate;
        double gen_rapr;
        int start = 0;
        CItemCorpo cit = new CItemCorpo(RULE_MAX_LENGHT);
        StringBuffer corpo = new StringBuffer();
        StringBuffer singolo_item = new StringBuffer();
        StringBuffer regola_max = new StringBuffer();

        int itemc = 0;

        if ( itemset == null ) {
            regole_rappresentate = gen_rapr = 1;
            corpo.append("{");

            for ( int i = 0 ; i<condition_lenght-1 ; i++) {
                corpo.append("(");
                for ( int j = 0 ; j<condition[i].numItem-1 ; j++ ) {
                    corpo.append(condition[i].vettItemId[j]+",");
                    cit.item_corpo[itemc] = condition[i].vettItemId[j];
                    itemc++;
                }
                corpo.append(condition[i].vettItemId[condition[i].numItem-1]+"),");
                cit.item_corpo[itemc] = condition[i].vettItemId[condition[i].numItem-1];
                itemc++;
                regole_rappresentate = regole_rappresentate * combinazioni(condition[i].numItem);
                gen_rapr = gen_rapr * condition[i].numItem;
            }

            corpo.append("(");
            for ( int k = 0 ; k<condition[condition_lenght-1].numItem-1 ; k++) {
                corpo.append(condition[condition_lenght-1].vettItemId[k]+",");
                cit.item_corpo[itemc] = condition[condition_lenght-1].vettItemId[k];
                itemc++;
            }
            corpo.append(condition[condition_lenght-1].vettItemId[condition[condition_lenght-1].numItem-1]+")}");
            cit.item_corpo[itemc] = condition[condition_lenght-1].vettItemId[condition[condition_lenght-1].numItem-1];
            itemc++;
            regole_rappresentate = regole_rappresentate * combinazioni(condition[condition_lenght-1].numItem);
            gen_rapr = gen_rapr * condition[condition_lenght-1].numItem;

            if ( dimcoda > 0 ) {
                for ( int h = 0 ; h<dimcoda-1 ; h++) {
                    corpo.append("(");
                    for ( int g = 0 ; g<coda[h].numItem-1 ; g++ ) {
                        corpo.append(coda[h].vettItemId[g]+",");
                        cit.item_corpo[itemc] = coda[h].vettItemId[g];
                        itemc++;
                    }
                    corpo.append(coda[h].vettItemId[coda[h].numItem-1]+"),");
                    cit.item_corpo[itemc] = coda[h].vettItemId[coda[h].numItem-1];
                    itemc++;
                    regole_rappresentate = regole_rappresentate * (combinazioni(coda[h].numItem)+1);

                }

                corpo.append("(");
                for ( int l = 0 ; l<coda[dimcoda-1].numItem-1 ; l++) {
                    corpo.append(coda[dimcoda-1].vettItemId[l]+",");
                    cit.item_corpo[itemc] = coda[dimcoda-1].vettItemId[l];
                    itemc++;
                }
                corpo.append(coda[dimcoda-1].vettItemId[coda[dimcoda-1].numItem-1]+")");
                cit.item_corpo[itemc] = coda[dimcoda-1].vettItemId[coda[dimcoda-1].numItem-1];
                itemc++;
                regole_rappresentate = regole_rappresentate * (combinazioni(coda[dimcoda-1].numItem)+1);

            } else {
                corpo.append("()");
            }
            cit.ordina(0,itemc-1);

            for ( int ic = 0 ; ic < itemc ; ic++) {
                regola_max.append(cit.item_corpo[ic]+" ");
            }

            if ( (items.suppClass[classeDesiderata] >= supp_thres) && ((double)(100.0 * (float)(items.suppClass[classeDesiderata])/(float)(items.supp)) >= CMain.conf_threshold) ) {
                DecimalFormat format = new DecimalFormat("###0.00",new DecimalFormatSymbols(new Locale("EN")));
                FieldPosition field = new FieldPosition(0);

                format.format(((100.0*items.suppClass[classeDesiderata])/items.supp),s,field);
                piw.append(corpo+" -> "+(CMain.idBaseClasse+classeDesiderata)+" "+items.suppClass[classeDesiderata]+" "+s+" "+itemc+" "+regola_max+"\n");
                piw.flush();
                corpo.delete(0, corpo.length());
                regola_max.delete(0, regola_max.length());
                s.delete(0, s.length());
                regoleTotali = regoleTotali+regole_rappresentate;

                macroRegoleTotali++;
            }

            if ( comb_size != 0) {
                itemset = new CMacroItem[comb_size];
                for ( int q = 0 ; q < comb_size ; q++ ) {
                    itemset[q] = new CMacroItem();
                }
                storeItemset(condition, condition_lenght, comb, comb_size, items, 0, 1, itemset, piw, supp_thres, coda, dimcoda);
            }

        }

        else {
            start = position;

            position = comb_size-1;

            while ( position >= start ) {
                itemset[store_level-1] = comb[position];
                regole_rappresentate = 1;
                gen_rapr = 1;
                corpo.append("{");

                for ( int i = 0 ; i<store_level ; i++) {
                    corpo.append("(");
                    for ( int j = 0 ; j<itemset[i].numItem-1 ; j++ ) {
                        corpo.append(itemset[i].vettItemId[j]+",");
                        cit.item_corpo[itemc] = itemset[i].vettItemId[j];
                        itemc++;
                    }
                    corpo.append(itemset[i].vettItemId[itemset[i].numItem-1]+"),");
                    cit.item_corpo[itemc] = itemset[i].vettItemId[itemset[i].numItem-1];
                    itemc++;
                    regole_rappresentate = regole_rappresentate*combinazioni(itemset[i].numItem);
                    gen_rapr = gen_rapr * itemset[i].numItem;
                }

                for ( int k= 0 ; k<condition_lenght-1 ; k++ ) {
                    corpo.append("(");
                    for ( int c = 0 ; c<condition[k].numItem-1; c++) {
                        corpo.append(condition[k].vettItemId[c]+",");
                        cit.item_corpo[itemc] = condition[k].vettItemId[c];
                        itemc++;
                    }
                    corpo.append(condition[k].vettItemId[condition[k].numItem-1]+"),");
                    cit.item_corpo[itemc] = condition[k].vettItemId[condition[k].numItem-1];
                    itemc++;
                    regole_rappresentate = regole_rappresentate*combinazioni(condition[k].numItem);
                    gen_rapr = gen_rapr*condition[k].numItem;
                }

                corpo.append("(");

                for ( int e = 0 ; e<condition[condition_lenght-1].numItem-1 ; e++ ) {
                    corpo.append(condition[condition_lenght-1].vettItemId[e]+",");
                    cit.item_corpo[itemc] = condition[condition_lenght-1].vettItemId[e];
                    itemc++;
                }
                corpo.append(condition[condition_lenght-1].vettItemId[condition[condition_lenght-1].numItem-1]+")}");
                cit.item_corpo[itemc] = condition[condition_lenght-1].vettItemId[condition[condition_lenght-1].numItem-1];
                itemc++;
                regole_rappresentate = regole_rappresentate*combinazioni(condition[condition_lenght-1].numItem);
                gen_rapr = gen_rapr * condition[condition_lenght-1].numItem;

                if ( dimcoda>0 ) {
                    for (int o=0 ; o<dimcoda-1 ; o++ ) {
                        corpo.append("(");
                        for ( int u=0 ; u<coda[o].numItem-1; u++ ) {
                            corpo.append(coda[o].vettItemId[u]+",");
                            cit.item_corpo[itemc] = coda[o].vettItemId[u];
                            itemc++;
                        }
                        corpo.append(coda[o].vettItemId[coda[o].numItem-1]+"),");
                        cit.item_corpo[itemc] = coda[o].vettItemId[coda[o].numItem-1];
                        itemc++;
                        regole_rappresentate = regole_rappresentate*(combinazioni(coda[o].numItem)+1);
                    }
                    corpo.append("(");
                    for ( int a=0 ; a<coda[dimcoda-1].numItem-1 ; a++ ) {
                        corpo.append(coda[dimcoda-1].vettItemId[a]+",");
                        cit.item_corpo[itemc] = coda[dimcoda-1].vettItemId[a];
                        itemc++;
                    }
                    corpo.append(coda[dimcoda-1].vettItemId[coda[dimcoda-1].numItem-1]+")");
                    cit.item_corpo[itemc] = coda[dimcoda-1].vettItemId[coda[dimcoda-1].numItem-1];
                    itemc++;
                    regole_rappresentate = regole_rappresentate*(combinazioni(coda[dimcoda-1].numItem)+1);

                } else {
                    corpo.append("()");
                }

                cit.ordina(0,itemc-1);
                for ( int ic = 0 ; ic<itemc ; ic++) {
                    singolo_item.append(cit.item_corpo[ic]+" ");
                }

                if ( (items.suppClass[classeDesiderata] >= supp_thres) && ((double)(100.0 * (float)(items.suppClass[classeDesiderata])/(float)(items.supp)) >= CMain.conf_threshold) ) {
                    DecimalFormat format = new DecimalFormat("###0.00",new DecimalFormatSymbols(new Locale("EN")));
                    FieldPosition field = new FieldPosition(0);
                    format.format(((100.0*items.suppClass[classeDesiderata])/items.supp),s,field);
                    piw.append(corpo+" -> "+(CMain.idBaseClasse+classeDesiderata)+" "+items.suppClass[classeDesiderata]+" "+s+" "+itemc+" "+singolo_item+"\n");//regola_max+"\n");
                    piw.flush();
                    corpo.delete(0, corpo.length());
                    singolo_item.delete(0, singolo_item.length());
                    s.delete(0, s.length());
                    regoleTotali = regoleTotali+regole_rappresentate;
                    macroRegoleTotali++;
                }

                storeItemset(condition, condition_lenght, comb, comb_size, items, position+1, store_level+1, itemset, piw, supp_thres, coda, dimcoda);
                position--;
            }

        }

    }


    public CHeaderTable headerTableCreate(CItemEntry tmptab,int supp_thres,int frequent_count) {
        CHeaderTable htab = new CHeaderTable(frequent_count);
        CItemEntry itemd;

        htab.frequentCount = frequent_count;

        if ( frequent_count != 0) {
            itemd = tmptab;
            int i = 0;
            while ( itemd != null ) {
                if ( itemd.suppClass[classeDesiderata] >= supp_thres ) {

                    htab.frequentArray[i].itemId = itemd.itemId;
                    htab.frequentArray[i].supp = itemd.supp;
                    htab.frequentArray[i].accorpato = 0;
                    htab.frequentArray[i].itemAccorpati = new LinkedList<Integer>();
                    for ( int b = 0 ; b < MAX_CLASSES ; b++) {
                        htab.frequentArray[i].suppClass[b] = itemd.suppClass[b];
                    }
                    htab.frequentArray[i].nodeLinkCounter = 0;
                    i++;
                }
                itemd = itemd.next;
            }
            htab.quicksort(0, frequent_count-1);
        }

        return htab;
    }


    public int accresciCodaperCondition(CMacroItem[] coda,int dimcoda,CHeaderTable htab, int suppcondition) {

        int ht;
        int sitem;
        CFrequentItem[] freq_array;

        freq_array = htab.frequentArray;

        for ( ht = 0 ; ht<htab.frequentCount ; ht++ ) {
            if ( freq_array[ht].supp == suppcondition ) {
                sitem = 1;
                coda[dimcoda].vettItemId[0] = freq_array[ht].itemId;
                Iterator<Integer> it = freq_array[ht].itemAccorpati.iterator();
                while ( it.hasNext()) {
                    coda[dimcoda].vettItemId[sitem] = it.next();
                    sitem++;
                }
                coda[dimcoda].numItem = sitem;
                dimcoda++;
            }
        }
        return dimcoda;
    }


    public CFptree fptCreate(CHeaderTable htab) {

        CFptree fp;
        CFpNode parent;
        CFpNode current;
        CFpNode nodelink;
        CFpNode pattern_node;
        CItem local = new CItem(MAX_CLASSES);
        CFrequentItem[] freq_array;
        int found;

        noditree = 0;

        fp = new CFptree(MAX_CLASSES);
        fp.branches = 0;

        noditree++;

        freq_array = htab.frequentArray;

        nodelink = node_pattern_base.head;

        int nlcounter = node_pattern_base.nodeLinkCounter;

        for ( int inl = 0 ; inl<nlcounter ; inl++) {
            local.supp = nodelink.local_supp;
            for ( int c = 0 ; c<MAX_CLASSES ; c++) {
                local.suppClass[c] = nodelink.local_suppClass[c];
            }
            parent = fp.root;

            for ( int ih = htab.frequentCount-1 ; ih>=0  ; ih--) {
                pattern_node = nodelink.parent;
                found = 0;
                while ( (found == 0) && (pattern_node!=null) ) {
                    if ( pattern_node.itemId == freq_array[ih].itemId )
                        found = 1;
                    else
                        pattern_node = pattern_node.parent;
                }
                if ( found == 1) {
                    if ( (current = insertNode(parent,freq_array[ih],local,fp)) == null ) {
                        return null;
                    }
                    noditree+=create_node;
                    parent = current;
                }
            }
            nodelink = nodelink.next;
        }

        return fp;
    }


    public CFpNode insertNode(CFpNode parent,CFrequentItem header_table_entry,CItem local,CFptree fp) {

        int found;
        CChildPtr curr_elem,prev_elem,tmp_elem;
        CFpNode tmp_child;

        found = 0;

        create_node = 0;

        curr_elem = parent.children;

        prev_elem = parent.children;

        while ( (curr_elem != null ) && (found == 0) ) {

            if ( curr_elem.child.itemId == header_table_entry.itemId) {
                found = 1;
            } else {
                prev_elem = curr_elem;
                curr_elem = curr_elem.next;
            }

        }

        if ( found == 1) {
            curr_elem.child.local_supp += local.supp;
            for ( int c = 0 ; c < MAX_CLASSES ; c++) {
                curr_elem.child.local_suppClass[c] += local.suppClass[c];
            }

            parent = curr_elem.child;
        }

        else {
            if ( (parent.children == null) && (parent.parent == null) ) {
                fp.setBranches(1);
            } else {
                if ( parent.children != null ) {
                    fp.setBranches(2);
                }
            }
            tmp_elem = new CChildPtr(MAX_CLASSES);
            create_node = 1;

            tmp_child = tmp_elem.child;
            tmp_child.itemId = header_table_entry.itemId;
            tmp_child.parent = parent;
            tmp_child.local_supp = local.supp;

            for ( int h = 0 ; h<MAX_CLASSES ; h++) {
                tmp_child.local_suppClass[h] = local.suppClass[h];
            }

            tmp_child.next = header_table_entry.head;


            if ( prev_elem == null ) {
                parent.children = tmp_elem;
            } else {
                prev_elem.next = tmp_elem ;
            }

            header_table_entry.incCounter(1);
            header_table_entry.head = tmp_child;

            parent = tmp_child;

        }

        return parent;
    }


    public double combinazioni( int numero ) {

        if ( numero < 1)
            return 0;

        double totale = 1;

        for ( int i = 0 ; i<numero ; i++) {
            totale = totale * 2;
        }
        return totale-1;
    }



}

