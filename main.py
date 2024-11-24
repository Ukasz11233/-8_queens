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
        if self.generation >= self.num_generations or self.best_score == 0:
            Clock.unschedule(self.run_generation)
            return

        population_fitness, total_num_attacks = self.fitness(self.population)

        max_fitness = numpy.max(population_fitness)
        max_fitness_idx = numpy.where(population_fitness == max_fitness)[0][0]

        self.best_outputs_fitness.append(max_fitness)
        self.best_outputs.append(self.population[max_fitness_idx])

        if total_num_attacks[max_fitness_idx] < self.best_score:
            self.best_score = total_num_attacks[max_fitness_idx]
            self.update_board_UI()

        if self.generation % 10 == 0:
            print("Generation = ", self.generation, "  Max Fitness = ", max_fitness, "  # Attacks = ",
                    total_num_attacks[max_fitness_idx])

        if max_fitness == float("inf"):
            print("Best solution found")
            self.num_attacks_Label.text = "Best Solution Found"
            print("\n**  Best solution IDX = ", max_fitness_idx, "  **\n")

            numpy.save("best_outputs_fitness.npy", self.best_outputs_fitness)
            numpy.save("best_outputs.npy", self.best_outputs)
            print("\n**  Data Saved Successfully  **\n")

            return

        parents = genAlg.select_parents(self.population, population_fitness, self.num_parents)
        offspring_crossover = genAlg.crossover(parents, offspring_size=(self.num_solutions - len(parents), N))
        offspring_mutation = genAlg.mutation(offspring_crossover, num_mutations=numpy.uint8(self.num_mutations_TextInput.text))

        self.population[0:len(parents)] = parents
        self.population[len(parents):] = offspring_mutation
        self.generation += 1


    def initialize_population(self, *args):
        self.num_solutions = numpy.uint8(self.num_solutions_TextInput.text)

        self.reset_board_text()

        self.population = [[(numpy.random.randint(0, N), numpy.random.randint(0, N)) for _ in range(N)] for _ in range(self.num_solutions)]

        self.pop_created = True
        self.num_attacks_Label.text = "Initial population Created."

    def fitness(self, population):
        total_num_attacks = numpy.zeros(len(population))

        for idx, solution in enumerate(population):
            total_num_attacks[idx] = self.calculate_attacks(solution)

        population_fitness = numpy.copy(total_num_attacks)
        for idx in range(len(population_fitness)):
            if population_fitness[idx] == 0:
                population_fitness[idx] = float("inf")
            else:
                population_fitness[idx] = 1.0 / population_fitness[idx]

        return population_fitness, total_num_attacks

    def calculate_attacks(self, solution):
        total_attacks = 0
        for i in range(N):
            for j in range(i + 1, N):
                if solution[i][0] == solution[j][0] or solution[i][1] == solution[j][1] or abs(solution[i][0] - solution[j][0]) == abs(solution[i][1] - solution[j][1]):
                    total_attacks += 1
        return total_attacks

    def reset_board_text(self):
        for row_idx in range(self.all_widgets.shape[0]):
            for col_idx in range(self.all_widgets.shape[1]):
                self.all_widgets[row_idx, col_idx].text = ""
                with self.all_widgets[row_idx, col_idx].canvas.before:
                    kivy.graphics.Color(0, 0, 0, 1)
                    self.rect = kivy.graphics.Rectangle(size=self.all_widgets[row_idx, col_idx].size, pos=self.all_widgets[row_idx, col_idx].pos)

    def update_board_UI(self, *args):
        if not self.pop_created:
            return
        
        def update_ui(*_):
            self.reset_board_text()
            print("Updating UI for generation = ", self.generation, "# Attacks = ", self.best_score)

            population_fitness, total_num_attacks = self.fitness(self.population)

            max_fitness = numpy.max(population_fitness)
            max_fitness_idx = numpy.where(population_fitness == max_fitness)[0][0]
            best_solution = self.population[max_fitness_idx]

            self.num_attacks_Label.text = f"Max Fitness = {max_fitness:.4f}\n# Attacks = {total_num_attacks[max_fitness_idx]}"

            for row_idx, col_idx in best_solution:
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
        boxLayout = kivy.uix.boxlayout.BoxLayout(orientation="vertical")

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
