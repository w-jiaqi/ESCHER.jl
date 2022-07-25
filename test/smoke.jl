@testset "smoke" begin
    game = Kuhn()
    sol = ESCHERSolver(game)
    cb = ESCHER.ExploitabilityCallback(sol)
    train!(sol, 10; cb=cb)

    vb = sol.value_buffer
    @test (vb[1] isa Tuple) && length(vb[1]) == 2
    @test_throws BoundsError vb[9_000_000]
    @test length(vb.x) == length(vb.y) == length(vb)
    @test length(vb) > 0

    rb1 = first(sol.regret_buffer)
    rb2 = last(sol.regret_buffer)
    @test length(rb1.x) == length(rb1.y) == length(rb1)
    @test length(rb1) > 0
    @test length(rb2.x) == length(rb2.y) == length(rb2)
    @test length(rb2) > 0

    @test length(sol.strategy_buffer.x) == length(sol.strategy_buffer.y)
    @test length(sol.strategy_buffer.x) > 0

    h0 = initialhist(game)
    @test ESCHER.value(sol, 1, vectorized_hist(game, h0)) isa Number

    ## GPU
    sol = ESCHERSolver(game; gpu=true)
    train!(sol, 10)
end
