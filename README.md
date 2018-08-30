
    Interview for Deep Learning Position
    
    Author: Jean Massucatto
    Computer Engineer 


    1 - First method

    Entendendo o problema de maneira que cada linha fosse um conjunto de 10 pontos, assim sendo, cada linha seria um próprio dataset a primeira solução implementa uma regressão linear utilizando o método Elastic Net sobre os 10 pontos do conjunto e percorrendo todo o conjunto maior com as 100k amostras.

    O método Elastic Net foi escolhido por ser um meio campo entre o método Lasso e o o método Ridge, possuindo uma penalidade L1 generalizar esparsidade e uma penalidade L2 para superar algumas das limitações do Lasso, como por exemplo o número de váriaveis selecionadas.

    Então a ideia mais simples foi transformar cada linha em um novo conjunto de dados, sendo um conjunto de duas novas colunas referentes as coordenadas dos 10 pontos de cada conjunto para então traçar a melhor linha entre eles.

    Após calculados os parametros de Slope e Intercept para o conjunto de treinamento e teste, um novo arquivo csv é criado para cada um dos resultados, sendo um arquivo de duas colunas contendo os parâmetros para todas as 100 amostras ou conjuntos até então.

    -> Compilando o primeiro método:

 	Foi criado um diretório chamado 'env' que irá conter todos os resultados da regressão, seja os resultados de treinamento como também o de teste, seguindo o exemplo descrito pelo GIST.

 	O algoritmo referente a este método está em 'elasticnet_regressor.py'.
 	Para a geração dos resultados de treino e teste, existem duas linhas que devem ser comentadas ou descomentadas referente ao seu devido conjunto de dados.
 		A primeira para selecionar o dataset desejado
 		A segunda para selecionar o arquivo a ser salvo.

 	OBS: Apesar dos bons resultados obtidos, os quais estão nos arquivos csv, acredito ter me equivocado no método de resolucão, portanto foi criada uma nova rede neural utilizando API Keras para realizar uma regressão no conjunto todo ao invés de amostra por amostra.

	

	2 - Second Method

	Após carregados os conjuntos de dados, foi criada uma rede neural utilizando API Keras juntamente com KerasRegressor implementando pela biblioteca sklearn para realizar uma regressão por todo o conjunto de dados tendo como entrada os 20 pontos iniciais de cada amostra do conjunto de treinamento maior, e não como outro conjunto de dados em cada uma das linhas como feito no problema acima.

	Por se tratar de uma regressão linear a ultima camada da rede, no caso a camada de saída implementa uma função de ativação linear enquanto que as camadas intermediárias podem utilizar as demais funções de ativação disponiveis, a função utilizada no algoritmo desenvolvido foi a selu.

	Após executado o algotirmo o procedimento é o mesmo já apresentado. Salvar os resultados da predição em um arquivo csv para ser utilizado mais tarde para calcular o MSE e MAE.

	O arquivo referente ao algoritmo desta rede neural encontra-se em 'nn_keras_regressor.py', bastando somente executa-lo para gerar as duas saídas de resultado.

    Os resultados da saída estão salvos no diretório 'env_keras', prontos para serem compilados para o cálculo dos erros seguindo o exemplo do GIST.