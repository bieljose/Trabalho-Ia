package trabalho;
//ctrl+shift+o faz imports automáticos
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class trabcarros {

	public static void main(String[] args) throws Exception {
		Instances carrosTreino;
		Instances carrosTeste;
		
		FileReader leitor = new FileReader("carrostrab.arff");
		Instances carros = new Instances(leitor);
		carros.setClassIndex(7); // define qual é o atributo classe
		
		carros = carros.resample(new Random());
		
		IB1 vizinho = new IB1();
		IBk knn = new IBk(3); //k = 3
		System.out.println("real;vizinho;knn");
		for (int j = 0; j<60;j++){
			
		carrosTreino = carros.trainCV(60,j); // 1º parametro = em quantas partes vai dividir a base. 2º parametro = iteração atual. 
		carrosTeste = carros.testCV(60, j);
		//-------------------------
		//treinando os classificadores
		vizinho.buildClassifier(carrosTreino);
		knn.buildClassifier(carrosTreino);
		
		//-------------------------
		//gravando os resultados em um arquivo .csv
		
		
		for(int i=0;i<carrosTeste.numInstances();i++){
			Instance teste = carrosTeste.instance(i);
			System.out.print(teste.value(7)+";");
			teste.setClassMissing();
			double vizinhoValue = vizinho.classifyInstance(teste);
			double knnvalue = knn.classifyInstance(teste);
			System.out.println(vizinhoValue+";"+knnvalue);
		}
		
		}
	}

}
