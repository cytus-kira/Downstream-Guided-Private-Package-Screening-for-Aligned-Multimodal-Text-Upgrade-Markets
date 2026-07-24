package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"ballmarket/vtd/internal/vtd"
)

func fail(err error) { fmt.Fprintln(os.Stderr, "[vtd]", err); os.Exit(1) }
func printJSON(v any) { b,_:=json.MarshalIndent(v,"","  "); fmt.Println(string(b)) }

func main() {
	if len(os.Args) < 2 { fail(fmt.Errorf("missing command")) }
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	root := fs.String("root", "run_vtd", "run root")
	role := fs.String("role", "", "buyer or market")
	preset := fs.String("preset", "production", "production or smoke")
	fixture := fs.String("fixture", "", "package fixture JSON")
	landmarks := fs.Int("landmarks", 1000, "quartic KRR landmarks")
	session := fs.String("session", "session-0", "session id")
	output := fs.String("output", "score-batch-0", "output id")
	permit := fs.Bool("permit", true, "release permission")
	attack := fs.String("attack", "malformed-share", "attack type")
	fs.Parse(os.Args[2:])
	var err error
	switch cmd {
	case "init":
		cfg := vtd.ProductionHEConfig(); if *preset=="smoke" { cfg=vtd.SmokeHEConfig() }
		err = vtd.InitRun(*root, cfg)
	case "party-round1": err = vtd.PartyRoundOne(*root, *role)
	case "coordinator-round1": err = vtd.CoordinatorRoundOne(*root)
	case "party-round2": err = vtd.PartyRoundTwo(*root, *role)
	case "coordinator-finalize": err = vtd.CoordinatorFinalize(*root)
	case "proof-setup":
		var m vtd.ProofMetrics; m,err=vtd.ProofSetup(*root); if err==nil { printJSON(m) }
	case "market-encrypt": err = vtd.MarketEncrypt(*root, *fixture)
	case "buyer-score":
		var m map[string]any; m,err=vtd.BuyerScore(*root,*landmarks); if err==nil { printJSON(m) }
	case "buyer-authorize": err = vtd.BuyerAuthorize(*root,*session,*output,*permit)
	case "market-release":
		var m vtd.ProofMetrics; m,err=vtd.MarketRelease(*root,vtd.ScorerDigest(*landmarks)); if err==nil { printJSON(m) }
	case "buyer-verify":
		var m vtd.BuyerReleaseResult; m,err=vtd.BuyerVerifyAndDecode(*root); if err==nil { printJSON(m) }
	case "attack-contribution": err = vtd.AttackModifyContribution(*root,*attack)
	default: err = fmt.Errorf("unknown command %q", cmd)
	}
	if err != nil { fail(err) }
}
