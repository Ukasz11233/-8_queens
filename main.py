import kivy.app
import kivy.uix.gridlayout
import kivy.uix.boxlayout
import kivy.uix.button
import kivy.uix.textinput
import kivy.uix.label
import kivy.graphics
import numpy
import genAlg
from kivy.clock import Clock

N = 8
class Queens8App(kivy.app.App):
    pop_created = False

    def start_ga(self, *args):
        self.initialize_population()
        self.best_outputs = []
        self.best_outputs_fitness = []
        if not self.pop_created:
            return

        self.num_generations = numpy.uint16(self.num_generations_TextInput.text)
        self.num_parents = numpy.uint8(self.num_solutions / 2)
        self.generation = 0
        self.best_score = 1000

        Clock.schedule_interval(self.run_generation, 0.1)

    def run_generation(self, dt):
        if self.generation>=self.num_generations or self.best_score == 0:
            Clock.unschedule(self.run_generation)
            return

        # print("\n##  Generation = ", generation, "  ##\n")

        population_fitness, total_num_attacks = self.fitness(self.population_board)

        max_fitness = numpy.max(population_fitness)
        max_fitness_idx = numpy.where(population_fitness == max_fitness)[0][0]

        self.best_outputs_fitness.append(max_fitness)
        self.best_outputs.append(self.population_1D_vector[max_fitness_idx, :])


        if total_num_attacks[max_fitness_idx] < self.best_score:
            self.best_score = total_num_attacks[max_fitness_idx]
            self.update_board_UI()
            


        if self.generation % 10 == 0:
            print("Generation = ", self.generation, "  Max Fitness = ", max_fitness, "  # Attacks = ",
                    total_num_attacks[max_fitness_idx])
        #     self.update_board_UI()
        #     # wait for 1 second to display the best solution
        #     kivy.clock.Clock.schedule_once(self.update_board_UI, 1)

        if max_fitness == float("inf"):
            print("Best solution found")
            self.num_attacks_Label.text = "Best Solution Found"
            print("\n1D population_board : \n", self.population_1D_vector)
            print("\n**  Best solution IDX = ", max_fitness_idx, "  **\n")

            numpy.save("best_outputs_fitness.npy", self.best_outputs_fitness)
            numpy.save("best_outputs.npy", self.best_outputs)
            print("\n**  Data Saved Successfully  **\n")

            return

        parents = genAlg.select_parents(self.population_1D_vector, population_fitness, self.num_parents)
        offspring_crossover = genAlg.crossover(parents, offspring_size=(self.num_solutions - parents.shape[0], N))
        offspring_mutation = genAlg.mutation(offspring_crossover, num_mutations=numpy.uint8(self.num_mutations_TextInput.text))

        self.population_1D_vector[0:parents.shape[0], :] = parents
        self.population_1D_vector[parents.shape[0]:, :] = offspring_mutation
        self.generation += 1

        self.vector_to_matrix()

    def initialize_population(self, *args):
        self.num_solutions = numpy.uint8(self.num_solutions_TextInput.text)

        self.reset_board_text()

        self.population_1D_vector = numpy.random.randint(0, N, size=(self.num_solutions, N))

        self.vector_to_matrix()

        self.pop_created = True
        self.num_attacks_Label.text = "Initial population_board Created."

    def vector_to_matrix(self):
        self.population_board = numpy.zeros(shape=(self.num_solutions, N, N))

        for solution_idx, current_solution in enumerate(self.population_1D_vector):
            current_solution = numpy.uint8(current_solution)
            for row_idx, col_idx in enumerate(current_solution):
                self.population_board[solution_idx, row_idx, col_idx] = 1

    def fitness(self, population_board):
        total_num_attacks_column = self.attacks_column(self.population_board)
        total_num_attacks_diagonal = self.attacks_diagonal(self.population_board)
        total_num_attacks_horizontal = self.attacks_horizontal(self.population_board)

        total_num_attacks = total_num_attacks_column + total_num_attacks_diagonal + total_num_attacks_horizontal

        population_fitness = numpy.copy(total_num_attacks)

        for solution_idx in range(population_board.shape[0]):
            if population_fitness[solution_idx] == 0:
                population_fitness[solution_idx] = float("inf")
            else:
                population_fitness[solution_idx] = 1.0 / population_fitness[solution_idx]

        return population_fitness, total_num_attacks

    def attacks_diagonal(self, population_board):

        total_num_attacks = numpy.zeros(population_board.shape[0])

        for solution_idx in range(population_board.shape[0]):
            ga_solution = population_board[solution_idx, :]

            temp = numpy.zeros(shape=(N+2, N+2))
            temp[1:N+1, 1:N+1] = ga_solution

            row_indices, col_indices = numpy.where(ga_solution == 1)
            row_indices += 1
            col_indices += 1

            total = 0
            for element_idx in range(N):
                x = row_indices[element_idx]
                y = col_indices[element_idx]

                total += self.diagonal_attacks(temp[x:, y:])  # Bottom-right
                total += self.diagonal_attacks(temp[x:, y:0:-1])  # Bottom-left
                total += self.diagonal_attacks(temp[x:0:-1, y:])  # Top-right
                total += self.diagonal_attacks(temp[x:0:-1, y:0:-1])  # Top-left

            total_num_attacks[solution_idx] += total / 2

        return total_num_attacks

    def diagonal_attacks(self, mat):

        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return 0
        return mat.diagonal().sum() - 1
    
    def attacks_column(self, population_board):

        total_num_attacks = numpy.zeros(population_board.shape[0])

        for solution_idx in range(population_board.shape[0]):
            ga_solution = population_board[solution_idx, :]

            for queen_y_pos in range(N):
                col_sum = numpy.sum(ga_solution[:, queen_y_pos])
                if col_sum > 1:
                    total_num_attacks[solution_idx] += col_sum - 1

        return total_num_attacks

    def attacks_horizontal(self, population_board):
        total_num_attacks = numpy.zeros(population_board.shape[0])

        for solution_idx in range(population_board.shape[0]):
            ga_solution = population_board[solution_idx, :]

            for queen_x_pos in range(N):
                row_sum = numpy.sum(ga_solution[queen_x_pos, :])
                if row_sum > 1:
                    total_num_attacks[solution_idx] += row_sum - 1

        return total_num_attacks

    def reset_board_text(self):
        for row_idx in range(self.all_widgets.shape[0]):
            for col_idx in range(self.all_widgets.shape[1]):
                # Ustawienie pustego tekstu dla wszystkich pól
                self.all_widgets[row_idx, col_idx].text = ""
                # Kolor tła pól pozostaje czarno-biały
                with self.all_widgets[row_idx, col_idx].canvas.before:
                    kivy.graphics.Color(0, 0, 0, 1)
                    self.rect = kivy.graphics.Rectangle(size=self.all_widgets[row_idx, col_idx].size,
                                                        pos=self.all_widgets[row_idx, col_idx].pos)

    def update_board_UI(self, *args):
        if not self.pop_created:
            return
        
        def update_ui(*_):
            self.reset_board_text()
            print("Updating UI for generation = ", self.generation, "# Attacks = ", self.best_score)

            population_fitness, total_num_attacks = self.fitness(self.population_board)

            max_fitness = numpy.max(population_fitness)
            max_fitness_idx = numpy.where(population_fitness == max_fitness)[0][0]
            best_solution = self.population_board[max_fitness_idx, :]

            self.num_attacks_Label.text = f"Max Fitness = {max_fitness:.4f}\n# Attacks = {total_num_attacks[max_fitness_idx]}"

            for row_idx in range(N):
                for col_idx in range(N):
                    if best_solution[row_idx, col_idx] == 1:
                        self.all_widgets[row_idx, col_idx].text = "w"
                        self.all_widgets[row_idx, col_idx].color = (1, 209/255, 0, 1)
                        with self.all_widgets[row_idx, col_idx].canvas.before:
                            kivy.graphics.Color(0, 1, 0, 1)
                            self.rect = kivy.graphics.Rectangle(
                                size=self.all_widgets[row_idx, col_idx].size,
                                pos=self.all_widgets[row_idx, col_idx].pos
                            )

        Clock.schedule_once(update_ui)

    def build(self):
        """
        Builds the graphical user interface (GUI) for the application,
        including the chessboard and control buttons for GA operations.
        """
        boxLayout = kivy.uix.boxlayout.BoxLayout(orientation="vertical" )

        # Grid layout for the chessboard
        gridLayout = kivy.uix.gridlayout.GridLayout(rows=N, cols=N, size_hint_y=80)
        boxLayout_buttons = kivy.uix.boxlayout.BoxLayout(orientation="horizontal", size_hint_y=N)

        boxLayout.add_widget(gridLayout)
        boxLayout.add_widget(boxLayout_buttons)

        # Initialize the chessboard widget array
        self.all_widgets = numpy.zeros(shape=(N, N), dtype="O")

        # Add buttons to the chessboard with alternating colors
        for row_idx in range(self.all_widgets.shape[0]):
            for col_idx in range(self.all_widgets.shape[1]):
                # Create a button for each square
                button = kivy.uix.button.Button(font_size=50, background_normal="", font_name="assets/chess.ttf", background_down="")
                # Alternate colors: white and black
                if (row_idx + col_idx) % 2 == 0:
                    button.background_color = (1, 1, 1, 1)  # White square
                else:
                    button.background_color = (0, 0, 0, 1)  # Black square

                # Add button to the grid and the widget array
                self.all_widgets[row_idx, col_idx] = button
                gridLayout.add_widget(button)

        # Buttons for Genetic Algorithm operations
        ga_solution_button = kivy.uix.button.Button(text="Show Best Solution", font_size=15, size_hint_x=2)
        ga_solution_button.bind(on_press=self.update_board_UI)

        start_ga_button = kivy.uix.button.Button(text="Start GA", font_size=15, size_hint_x=2)
        start_ga_button.bind(on_press=self.start_ga)

        # Inputs for GA parameters
        self.num_solutions_TextInput = kivy.uix.textinput.TextInput(text="250", font_size=20, size_hint_x=1)
        self.num_generations_TextInput = kivy.uix.textinput.TextInput(text="500", font_size=20, size_hint_x=1)
        self.num_mutations_TextInput = kivy.uix.textinput.TextInput(text="5", font_size=20, size_hint_x=1)

        # Display for fitness/attack stats
        self.num_attacks_Label = kivy.uix.label.Label(text="# Attacks/Best Solution", font_size=15, size_hint_x=2)

        # Add buttons and inputs to the layout
        boxLayout_buttons.add_widget(ga_solution_button)
        boxLayout_buttons.add_widget(start_ga_button)
        boxLayout_buttons.add_widget(self.num_solutions_TextInput)
        boxLayout_buttons.add_widget(self.num_generations_TextInput)
        boxLayout_buttons.add_widget(self.num_mutations_TextInput)
        boxLayout_buttons.add_widget(self.num_attacks_Label)

        return boxLayout

from kivy.core.window import Window
Window.size = (800, 880)

queens = Queens8App()
queens.run()
