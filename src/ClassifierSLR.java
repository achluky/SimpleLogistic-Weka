import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

import weka.classifiers.functions.SimpleLogistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
/**
 * 
 * 
 * @author ahmadluky
 *
 */
public class ClassifierSLR {

	Instances instances;
	SimpleLogistic classifier;
	public void loadModel(String fileName) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
			classifier = (SimpleLogistic) tmp;
            in.close();
 			System.out.println("===== Loaded model: " + fileName + " =====");
       } 
		catch (Exception e) {
			// Given the cast, a ClassNotFoundException must be caught along with the IOException
			System.out.println("Problem found when reading: " + fileName);
		}
	}
	
	@SuppressWarnings("unchecked")
	public void makeInstance(String csvInstance) throws NumberFormatException, IOException {
		
		//@RELATION iris
		//@ATTRIBUTE sepallength	REAL
		//@ATTRIBUTE sepalwidth 	REAL
		//@ATTRIBUTE petallength 	REAL
		//@ATTRIBUTE petalwidth		REAL
		//@ATTRIBUTE class 			{Iris-setosa,Iris-versicolor,Iris-virginica}
		
		// Create the header
		@SuppressWarnings("rawtypes")
		ArrayList  attributeList = new ArrayList(5);
		
		// Atribute "sepallength" - default numeric
		Attribute attribute = new Attribute("sepallength");
		attributeList.add(attribute);

		// Atribute "sepalwidth" - default numeric
		attribute = new Attribute("sepalwidth");
		attributeList.add(attribute);

		// Atribute "petallength" - default numeric
		attribute = new Attribute("petallength");
		attributeList.add(attribute);

		// Atribute "petalwidth" - default numeric
		attribute = new Attribute("petalwidth");
		attributeList.add(attribute);
		
		// Atribute "class"
		@SuppressWarnings("rawtypes")
		ArrayList  values = new ArrayList(3); 
		values.add("Iris-setosa"); 
		values.add("Iris-versicolor"); 
		values.add("Iris-virginica"); 
		attribute = new Attribute("class", values);
		attributeList.add(attribute);
		
		instances = new Instances("Test relation", (java.util.ArrayList<Attribute>) attributeList, 1);
		instances.setClassIndex(instances.numAttributes()-1);
		
		DenseInstance instance = new DenseInstance(5);
		instance.setDataset(instances);

		@SuppressWarnings("resource")
		BufferedReader reader = new BufferedReader(new FileReader(csvInstance));
		String line;
		while((line=reader.readLine()) != null){
			String[] stringValues = line.split(",");
			instance.setValue(0, Double.parseDouble(stringValues[0]));
			instance.setValue(1, Double.parseDouble(stringValues[1]));
			instance.setValue(2, Double.parseDouble(stringValues[2]));
			instance.setValue(3, Double.parseDouble(stringValues[3]));		
			instances.add(instance);
		}
 		System.out.println("===== Instance created with reference dataset =====");
		System.out.println(instances);
	}
	
	public void classify(Integer count) {
		try {
			System.out.println("===== Classified instance =====");
			for(int i = 0; i<count;i++){
				double pred = classifier.classifyInstance(instances.instance(i));
				System.out.println("Class predicted: " + instances.classAttribute().value((int) pred));
			}
		}
		catch (Exception e) {
			System.out.println("Problem found when classifying the example");
		}		
	}
	
	/**
	 * Main method
	 * @throws IOException 
	 * @throws NumberFormatException 
	 */
	public static void main (String[] args) throws NumberFormatException, IOException {
	
		ClassifierSLR classifier;

		String datT 	= "./test.csv";
		String modl 	= "./model-iris.dat";
		int count_datT 	= 9;
		classifier		= new ClassifierSLR();
		classifier.loadModel(modl);
		classifier.makeInstance(datT);
		classifier.classify(count_datT);
		
	}
}
