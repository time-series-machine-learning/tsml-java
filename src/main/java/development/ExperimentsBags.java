
package development;

import java.io.File;
import utilities.ClassifierTools;
import weka.core.Instances;

/**
 * The main experiments class for the bags project - main difference is the sampling method 
 * which assumes a leave-one-bag-out sampling system
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ExperimentsBags {//extends Experiments {
    
    public static void main(String[] args) throws Exception{
        Experiments.useBagsSampling = true;
        
        //exampleusage
//        Experiments.main(new String[] { 
//            "Z:\\Bags\\Classification Results\\Histogram Classification on Pre-Segmented GT\\",
//            "Z:\\Bags\\Classification Results\\testtesttest\\",
//            "true",
//            "ED",
//            "BagsTwoClassHistogramProblem",
//            "45"
//        });
        
        
        Experiments.main(args);
    }
    
    public static int[][] bagIndices = {																							
        /*Bag1*/{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,							},
        /*Bag2*/{ 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,							},
        /*Bag3*/{ 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,							},
        /*Bag4*/{ 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,						},
        /*Bag5*/{ 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 								},
        /*Bag6*/{ 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,                           	},
        /*Bag7*/{ 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 							},
        /*Bag8*/{ 93, 94, 95, 96, 97, 98, 99, 100,101,102,103,104,105,							},
        /*Bag9*/{ 106,107,108,109,110,111,112,113,114,115,116,117,              					},
        /*Bag10*/{118,119,120,121,122,123,124,125,126,127,128,129,130,131,						},
        /*Bag11*/{132,133,134,135,136,137,138,139,140,141,142,								},
        /*Bag12*/{143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,					},
        /*Bag13*/{160,161,162,163,164,165,166,167,168,169,170,171,172,173,						},
        /*Bag14*/{174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,				},
        /*Bag15*/{193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,					},
        /*Bag16*/{210,211,212,213,214,215,216,217,218,219,220,221,222,  						},
        /*Bag17*/{223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,	},
        /*Bag18*/{247,248,249,250,251,252,253,254,255,256,257,258,259,							},
        /*Bag19*/{260,261,262,263,264,265,266,267,268,269,270,271,272,273,						},
        /*Bag20*/{274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,					},
        /*Bag21*/{291,292,293,294,295,296,297,298,299,300,301,302,303,							},
        /*Bag22*/{304,305,306,307,308,309,310,311,312,313,314,315,316,317,      					},
        /*Bag23*/{318,319,320,321,322,323,324,325,326,327,328,329,330,                  				},
        /*Bag24*/{331,332,333,334,335,336,337,338,									},
        /*Bag25*/{339,340,341,342,343,344,345,346,									},
        /*Bag26*/{347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,					},
        /*Bag27*/{363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,					},
        /*Bag28*/{379,380,381,382,383,384,385,386,387,388,								},
        /*Bag29*/{389,390,391,392,393,394,395,396,397,398,								},
        /*Bag30*/{399,400,401,402,403,404,405,406,407,408,								},
        /*Bag31*/{409,410,411,412,413,414,415,416,417,418,419,420,							},
        /*Bag32*/{421,422,423,424,425,426,427,428,429,430,431,432,433,							},
        /*Bag33*/{434,435,436,437,438,439,440,441,442,443,444,445,							},
        /*Bag34*/{446,447,448,449,450,451,452,453,454,455,456,457,458,459,                                              },
        /*Bag35*/{460,461,462,463,464,465,466,467,468,469,470,471,                                              	},
        /*Bag36*/{472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,              		},
        /*Bag37*/{490,491,492,493,494,495,496,497,498,499,500,501,502,                          			},
        /*Bag38*/{503,504,505,506,507,508,509,510,511,512,513,514,515,                  				},
        /*Bag39*/{516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,  			},
        /*Bag40*/{535,536,537,538,539,540,541,542,543,544,545,546,547,							},
        /*Bag41*/{548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,      },
        /*Bag42*/{572,573,574,575,576,577,578,579,580,581,582,583,584,							},
        /*Bag43*/{585,586,587,588,589,590,591,592,593,594,595,596,597,598,						},
        /*Bag44*/{599,600,601,602,603,604,605,606,607,608,								},
        /*Bag45*/{609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,					},
    };           
    public static Instances[] sampleDataset(String problem, int fold) throws Exception {        
        if (fold < 0 || fold >= bagIndices.length)
            throw new Exception("[ExperimentsBagsLOOCV.sampleDataset] Given foldid greater than number of bags to sample from, fold="+fold+", numBags="+bagIndices.length);
        
        Instances[] data = new Instances[2];
        
        File trainFile = new File(DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN.arff");
        File testFile = new File(DataSets.problemPath+problem+"/"+problem+fold+"_TEST.arff");
        if(trainFile.exists() && testFile.exists()) {
            data[0] = ClassifierTools.loadData(trainFile.getAbsolutePath());
            data[1] = ClassifierTools.loadData(testFile.getAbsolutePath());
        }
        else { //make the folds from the full dataset
            int[] testInds = bagIndices[fold];

            Instances all = ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
            data[1] = new Instances(all, 0);

            //todo should technically sort in descending and iterate normally
            //go from back to front over the inds so i dont need to worry about indices changing as i remove elements
            //but always adding to front of test insts, so that object order is maintained
            for (int i = testInds.length-1; i >= 0; i--)
                data[1].add(0, all.remove(testInds[i] - 1)); //indexing from 1 with the row ids, -1 to fix
            data[0] = all;
        }
            
        return data;
    }
}
