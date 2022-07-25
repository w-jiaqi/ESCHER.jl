@testset "fitting" begin
    game = Kuhn()
    sol = ESCHERSolver(game)
    ESCHER.traverse_value!(sol)
    tr = ESCHER.training_run(sol.value, sol.value_buffer, sol.value_batch_size, sol.value_batches, Adam(1e-3))
    @test length(tr.loss) == sol.value_batches
    @test tr.lower_limit ≥ 0.
    @test all(≥(tr.lower_limit), tr.loss)
end
