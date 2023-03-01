# ------ GA Programming -----
# 00000 00000부터 11111 11111까지 가장 큰 이진 정수를 GA로 찾기
# 탐색 중에 해집단의 해들이 일정 비율 동일하게 수렴하면 최적 해로 수렴했다고 판단하고 탐색을 종료하도록 설계
# ---------------------------

# ----- 제약사항 ------
# pandas 모듈 사용 금지
# random 모듈만 사용, 필요시 numpy 사용 가능
# [chromosome, fitness]로 구성된 list 타입의 해 사용: ["1010", 10]
# population 형태는 다음과 같이 list 타입으로 규정: [["1010", 10], ["0001", 1], ["0011", 3]]
# --------------------

import random
import numpy as np
# ----- 수정 가능한 파라미터 -----

params = {
    'MUT': 0.5,  # 변이확률(%)
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE' : 10,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'UNIT' : 2,  # 8진수까지 표현가능
    'SELECTION_PRESSURE' : 3.
    # 원하는 파라미터는 여기에 삽입할 것
    }
# ------------------------------

class GA():
    def __init__(self, parameters):
        self.params = parameters
        self.population = None

    def get_fitness(self, chromosome):
        # todo: 이진수 -> 십진수로 변환하여 fitness 구하기
        fitness = 0
        new_cromosome = ''
        for i in range(len(chromosome)):
            fitness += int(chromosome[-1-i])*(self.params['UNIT']**i)
            new_cromosome += str(chromosome[i])
        chromosome_fit = [new_cromosome, fitness]
        return chromosome_fit

    def print_average_fitness(self):
        # todo: population의 평균 fitness를 출력
        population_average_fitness = 0
        for i in range(len(self.population)):
            population_average_fitness += self.population[i][1]
        print(population_average_fitness/len(self.population))


    def sort_population(self):
        self.population.sort(key=lambda x: x[1], reverse=True)
        # todo: fitness를 기준으로 population을 내림차순 정렬하고 반환


    def selection_operater(self):
        # todo: 본인이 원하는 선택연산 구현(룰렛휠, 토너먼트, 순위 등), 선택압을 고려할 것, 한 쌍의 부모 chromosome 반환
        sum_fit = 0
        scan_num = 0
        i = 0
        j = 0
        for h in range(len(self.population)):
            sum_fit += self.population[h][1]
        temp = np.random.randint(sum_fit, size=2)

        while(1):
            scan_num += self.population[i][1]
            if scan_num >= temp[0]:
                mom_ch = self.population[i]
                break
            i += 1
        scan_num = 0
        while(1):
            scan_num += self.population[i][1]
            if scan_num >= temp[1]:
                dad_ch = self.population[i]
                break
            j += 1
        return mom_ch, dad_ch




    def crossover_operater(self, mom_cho, dad_cho):
        # todo: 본인이 원하는 교차연산 구현(point, pmx 등), 자식해 반환
        number = np.random.randint(self.params['RANGE'], size=2)
        number.sort()
        offspring = mom_cho[:number[0]] + dad_cho[number[0]:number[1]] + mom_cho[number[1]:]
        return offspring


    def mutation_operater(self, chromosome):
        # todo: 변이가 결정되었다면 chromosome 안에서 랜덤하게 지정된 하나의 gene를 반대의 값(0->1, 1->0)으로 변이
        i = random.randrange(0, self.params['RANGE'])
        gene = int(chromosome[0][i]) + 1
        if gene == self.params['UNIT']:
            gene = 0
        chromosome[0] = chromosome[0][:i] +str(gene) + chromosome[0][i+1:]
        return chromosome

    def replacement_operator(self, offsprings):
        # todo: 생성된 자식해들(offsprings)을 이용하여 기존 해집단(population)의 해를 대치하여 새로운 해집단을 return
        new_population = self.population[:self.params["NUM_OFFSPRING"]] + offsprings
        return new_population

    # 해 탐색(GA) 함수
    def search(self):
        generation = 0  # 현재 세대 수
        offsprings = [] # 자식해집단

        # 1. 초기화: 랜덤하게 해를 초기화
        # 해집단
        # todo: random 모듈을 사용하여 랜덤한 해 생성, self.params["range"]를 사용할 것
        # todo: fitness를 구하는 함수인 self.get_fitness()를 만들어서 fitness를 구할 것
        self.population = [self.get_fitness(np.random.randint(self.params['UNIT'], size=self.params["RANGE"]))
                           for i in range(self.params['POP_SIZE'])]
        # todo: 정렬함수인 self.sort_population()을 사용하여 population을 정렬할 것
        self.sort_population()

        for i in range(len(self.population)):
            print(f"initialzed population : \n {self.population[i][0]}\n")

        while 1:
            offsprings = []
            for i in range(self.params["NUM_OFFSPRING"]):
                # 2. 선택 연산
                mom_ch, dad_ch = self.selection_operater()

                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                # todo: 변이 연산여부를 결정, self.params["MUT"]에 따라 변이가 결정되지 않으면 변이연산 수행하지 않음
                mut_num = random.randrange(1, round(100 / self.params['MUT']))
                if mut_num == 1:
                    offspring = self.mutation_operater(offspring)

                self.get_fitness(offspring)
                offsprings.append(offspring)

            self.sort_population()
            # 5. 대치 연산
            self.population = self.replacement_operator(offsprings)

            self.print_average_fitness() # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능

            print(self.population)
            # 6. 알고리즘 종료 조건 판단
            # todo population이 전체 중 self.params["END"]의 비율만큼 동일한 해를 갖는다면 수렴했다고 판단하고 탐색 종료
            if (self.params['END']*10 == self.population.count(self.population[0])): break
            else: pass

        # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
        print(f"탐색이 완료되었습니다. \t 최종 세대수: {generation},\t 최종 해: {self.population[0][0]},\t 최종 적합도: {self.population[0][1]}")


if __name__ == "__main__":
    ga = GA(params)
    ga.search()


