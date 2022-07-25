@testset "value" begin
    N = 100_000
    game = Kuhn()
    sol = ESCHERSolver(game)
    h0 = initialhist(game)
    v̂₀ = sum(ESCHER.value_traverse(sol, h0) for _ in 1:N)/N
    v₀ = ESCHER.exact_value(sol, h0)
    @test isapprox(v̂₀, v₀, atol=3e-2)

    for a in chance_actions(game, h0)
        h = next_hist(game, h0, a)
        v̂ = sum(ESCHER.value_traverse(sol, h) for _ in 1:N)/N
        v = ESCHER.exact_value(sol, h)
        @test isapprox(v̂,v, atol=3e-2)
    end
end
